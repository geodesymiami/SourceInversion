# SourceInversion

Add VSM to pythonpath (we haven't added anything on the environment.bash yet)
```
export PYTHONPATH=$$RSMASINSAR_HOME/tools/VSM/VSM:$PYTHONPATH
```

## Segguested
### Run each step separatedly

Downsample
```
src/cli/run_downsample --folder Chiles --satellite Sen --period=20220531:20220930 --method uniform --show
```

Inversion
```
src/cli/run_inversion --folder Chiles --satellite Sen --period=20220531:20220930 --show --model mogi
```

## To test
### Run alltogether

Each step has its own arguments defined in the [template](template.json)

I.e.:
```
{
    "decomposition": "--folder CampiFlegrei --satellite Sen --method uniform --show",
    "inversion": "--folder CampiFlegrei --satellite Sen --model mogi --show"
}
```
Run the command

```
src/cli/run_all.py
```
