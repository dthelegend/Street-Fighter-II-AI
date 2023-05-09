# Street-Fighter-II-AI

## Install

```bash
conda env create -f environment.yml
```

```bash
poetry install
```

```bash
python3 -m retro.import roms/
```

# Run
```bash
poetry run main
```

## Notes

Controls are mapped to the buttons using a binary format

| 128 | 64 | 32 | 16 | - | 8 | 4 | 2 | 1 |
| --- | -- | -- | -- | - | - | - | - | - |
|  -  |  C |  B |  A | - | - | Z | Y | X |

Characters are mapped by the following key:
- 00 – Ryu
- 01 – E. Honda
- 02 – Blanka
- 03 – Guile
- 04 – Ken
- 05 – Chun Li
- 06 – Zangief
- 07 – Dhalsim
- 08 – M.Bison
- 09 – Sagat
- 0A – Balrog
- 0B – Vega
