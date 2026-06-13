# Wave_local

### General

Build project:

```
git clone https://github.com/rs8055/Wave_local.git
cd Wave_local/combined
mkdir build
cd build
cmake .. -DDEAL_II_DIR=/PATH/TO/THE/DEAL/II/BUILD/FOLDER
make -j 4
```

Switch to release mode:

```
make release
```

Switch back to debug mode:

```
make debug
```

Run wave application:

```
./combined
```

