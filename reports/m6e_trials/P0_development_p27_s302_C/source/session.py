"""Exclusive tick-owner session with best-effort cleanup on every exit path."""
import warnings


class CarlaSession:
    def __init__(self, config, *, load_map=False):
        self.config = config
        self.load_map = load_map
        self.actors = []
        self.sensors = []
        self.original = None
        self.original_weather = None
        self.world = None
        self.traffic_manager = None
        self.autopilot_ego = None
        self.cleanup_callbacks = []

    def __enter__(self):
        import carla
        self.client = carla.Client(self.config['host'], self.config['port'])
        self.client.set_timeout(self.config['client_timeout_s'])
        if (self.client.get_client_version(), self.client.get_server_version()) != ('0.9.15', '0.9.15'):
            raise RuntimeError('Both CARLA client and server must be exactly 0.9.15')
        self.world = self.client.get_world()
        if self.world.get_settings().synchronous_mode:
            raise RuntimeError('World already synchronous; establish exclusive tick ownership first')
        requested = self.config['map']
        if self.world.get_map().name.split('/')[-1] != requested:
            if not self.load_map:
                raise RuntimeError(f'Expected {requested}; use --load-map on a dedicated server')
            self.world = self.client.load_world(requested)
        self.original = self.world.get_settings()
        self.original_weather = self.world.get_weather() if hasattr(self.world,'get_weather') else None
        # Boost.Python WorldSettings is not copyable/pickleable. Each API call
        # returns an independent settings value, leaving original intact.
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config['fixed_delta_seconds']
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        try:
            self.world.apply_settings(settings)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def spawn(self, blueprint, transform, *, parent=None):
        import carla
        kwargs = {} if parent is None else {'attach_to': parent, 'attachment_type': carla.AttachmentType.Rigid}
        actor = self.world.spawn_actor(blueprint, transform, **kwargs)
        self.actors.append(actor)
        if actor.type_id.startswith('sensor.'):
            self.sensors.append(actor)
        return actor

    def start_autopilot(self, ego, *, port, speed_mps):
        """Own a new TM on an unused port; never modify an existing TM instance."""
        import socket
        with socket.socket() as probe:
            if probe.connect_ex((self.config['host'], port)) == 0:
                raise RuntimeError('Traffic Manager port already in use; choose an exclusive port')
        self.traffic_manager = self.client.get_trafficmanager(port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.config['seed'])
        self.traffic_manager.auto_lane_change(ego, False)
        self.traffic_manager.set_desired_speed(ego, speed_mps * 3.6)
        import carla
        ego.apply_control(carla.VehicleControl())
        self.autopilot_ego = ego
        ego.set_autopilot(True, port)

    def __exit__(self, *_):
        for callback in reversed(self.cleanup_callbacks):
            try:callback()
            except Exception as exc:warnings.warn(f'Supplement cleanup failed: {exc}')
        self.cleanup_callbacks.clear()
        if self.traffic_manager is not None:
            try:
                if self.autopilot_ego is not None:
                    self.autopilot_ego.set_autopilot(False, self.traffic_manager.get_port())
            except Exception as exc:
                warnings.warn(f'Autopilot cleanup failed: {exc}')
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception as exc:
                warnings.warn(f'Traffic Manager cleanup failed: {exc}')
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception as exc:
                warnings.warn(f'sensor.stop failed: {exc}')
        for actor in reversed(self.actors):
            try:
                if not actor.destroy():
                    warnings.warn(f'Actor {actor.id} destruction returned false')
            except Exception as exc:
                warnings.warn(f'actor.destroy failed: {exc}')
        if self.original_weather is not None:
            try:
                self.world.set_weather(self.original_weather)
            except Exception as exc:
                warnings.warn(f'Weather restoration failed: {exc}')
        if self.original is not None:
            try:
                self.world.apply_settings(self.original)
            except Exception as exc:
                warnings.warn(f'World settings restoration failed: {exc}')
        self.actors.clear()
        self.sensors.clear()
