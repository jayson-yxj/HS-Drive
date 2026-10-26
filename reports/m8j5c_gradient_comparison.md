# 整网梯度三组对照

|精度/模式|对照|最大绝对误差|最大相对L2误差|最小cosine|容差结果|
|---|---|---:|---:|---:|---|
|FP32_eval|original_original|6.03497028e-07|1.06972128e-06|0.999999999999|Passed|
|FP32_eval|new_new|5.77419996e-07|1.04820653e-06|0.999999999999|Passed|
|FP32_eval|original_new|6.30505383e-07|1.09311862e-06|0.999999999999|Passed|
|FP32_train|original_original|5.81145287e-07|9.67453155e-07|1.000000000000|Passed|
|FP32_train|new_new|5.66244125e-07|9.18593632e-07|1.000000000000|Passed|
|FP32_train|original_new|7.89761543e-07|9.20446079e-07|1.000000000000|Passed|
|OfficialAMP_eval|original_original|0.0009765625|0.00098025711|0.999999519564|Failed|
|OfficialAMP_eval|new_new|0.0009765625|0.000978210952|0.999999521927|Failed|
|OfficialAMP_eval|original_new|0.0009765625|0.000981699584|0.999999518150|Failed|
|OfficialAMP_train|original_original|0.00048828125|0.000935304267|0.999999562622|Failed|
|OfficialAMP_train|new_new|0.00048828125|0.000933126697|0.999999565990|Failed|
|OfficialAMP_train|original_new|0.00048828125|0.000945178556|0.999999555935|Failed|
