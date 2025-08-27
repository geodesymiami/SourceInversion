# SourceInversion
## Define `SOURCEINVERSION_HOME` folder

```
export SOURCEINVERSION_HOME=path/to/code/SourceInversion
```

Add to pythonpath (we haven't added anything on the environment.bash yet)
```
export PATH=${SOURCEINVERSION_HOME}/src/cli:$PATH
export PYTHONPATH=${SOURCEINVERSION_HOME}/src:$PYTHONPATH
```

## Suggested
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
    "downsample": "--folder CampiFlegrei --satellite Sen --method uniform --show",
    "inversion": "--folder CampiFlegrei --satellite Sen --model mogi --show"
}
```
Run the command

```
src/cli/run_all.py
```
