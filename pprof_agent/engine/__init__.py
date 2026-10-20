"""The engine: the closed set of operations that produce numbers.

`rbridge`   runs one R script (pprofR, grplasso, the ported empirical null) through a JSON handshake
`executor`  maps an action of the closed set to its script and payload, keeps every payload and output
            in the run directory (what `repro.R` replays), and turns the output into a `Result`
`pycore`    the pprof_py presentation layer: profiles built from the R tables, figures, the HTML report
"""
