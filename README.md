# SourceInversion
## Define `SOURCEINVERSION_HOME` folder

```bash
cd SourceInversion
export SOURCEINVERSION_HOME=${PWD}
```

Add to pythonpath (we haven't added anything on the environment.bash yet)
```bash
export PATH=${SOURCEINVERSION_HOME}/src/cli:$PATH
export PYTHONPATH=${SOURCEINVERSION_HOME}/src:$PYTHONPATH
```

## Clone VSM and initialize as a package
```bash
git clone https://github.com/EliTras/VSM.git ${SOURCEINVERSION_HOME}/src/VSM
touch ${SOURCEINVERSION_HOME}/src/VSM/__init__.py
```

## Suggested
### Run each step separatedly

Downsample
```bash
run_downsample --folder Chiles --satellite Sen --period=20220531:20220930 --method uniform --show
```

Inversion
```bash
run_inversion --folder Chiles --satellite Sen --period=20220531:20220930 --show --model mogi
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

```bash
run_all.py
```
