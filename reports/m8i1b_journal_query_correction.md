# Journal 窗口查询勘误

最终核对stderr时发现，首批三个journal窗口查询不接受ISO形式时间参数，返回码非零。那些失败已保留在原始命令记录，**不能作为“窗口无事件”的证据**。`journal_boots` 独立命令已明确报告没有journal文件；内核dmesg的UTC窗口过滤成功，二者不得混淆。

本轮只追加修正后的只读命令，使用 `YYYY-MM-DD HH:MM:SS UTC`，分别查询相同H1/I1/I1a窗口。三条命令均rc0，stdout为 `-- No entries --`，stderr为 `No journal files were found.`。正确结论仍是**journal存储不可用，无法从journal判断是否发生系统异常**，不是已完整审查journal后排除错误。

原错误查询、不成功状态与原封存文件均不覆盖。修正记录见[m8i1b_journal_query_correction.json](m8i1b_journal_query_correction.json)，补充SHA见[m8i1b_supplement_manifest.json](m8i1b_supplement_manifest.json)。根因决策B保持不变，没有任何CARLA连接、启动或参数修改。
