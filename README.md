# arcnca

arc-2024 with nca

- https://arcprize.org/
- https://www.kaggle.com/competitions/arc-prize-2024

- https://github.com/maxencefaldor/cax/blob/main/examples/1d_arc_nca.ipynb


various types of compute were used

- `ojo` - a local nvidia agx orin
- `oop` - a local linux pc with nvidia 3090
- `big` - a cloud instance with 1xH100
- `kag` - a cloud kaggle nb with 4xT4

to run the test on a machine, ssh into it:

```bash
git clone https://github.com/hu-po/arcnca
./scripts/test.big.sh
```

for kaggle you need to "create a notebook" from the ["code" page](https://www.kaggle.com/competitions/arc-prize-2024/code) and paste in the contents of `test.ipynb` then click "save version" and make sure to disable internet. then go to the ["submit" page](https://www.kaggle.com/competitions/arc-prize-2024/submit) and hit "submit prediction".