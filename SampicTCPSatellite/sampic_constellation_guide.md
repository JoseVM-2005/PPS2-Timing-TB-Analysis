# SAMPIC + Constellation — Full Run Guide

This guide walks through every step from installation to a complete acquisition run.
Two terminals are needed: one for the **satellite** (the SAMPIC driver), one for the **controller** (the operator console).

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | ≥ 3.10 |
| SAMPIC GUI | Running on the Windows PC, TCP server active on port 8261 |
| Network | Your Linux/Mac machine can reach `192.168.1.46:8261` (ping it first) |

---

## Step 1 — Install Constellation

Run this once in the .venv environment where you will execute both the satellite and the controller.

```bash
# Core framework + the satellite base class
pip install ConstellationDAQ

# CLI controller (IPython-based interactive console)
pip install "ConstellationDAQ[cli]"
```

Verify the install:

```bash
python -c "import constellation; print('OK')"
Controller --help
```

---

## Step 2 — Create the configuration file

Save the following as **`sampic_config.toml`** in your working directory.
Edit the values to match your setup.

```toml
# sampic_config.toml
# -------------------
# Settings sent to the SAMPIC satellite on initialize.

[Sampic.One]
sampic_ip       = "192.168.1.46"   # IP of the Windows PC running SAMPIC GUI
sampic_port     = 8261             # TCP port of the SAMPIC server

save_directory  = 'C:\SAMPICdata'  # Folder on the WINDOWS machine where data is saved
base_filename   = "testrun"        # Prefix for output files

run_time_s      = 0                # Seconds to run (0 = continuous until stop command)
run_hits        = 0                # Stop after N hits (0 = disabled)

# setup_file    = 'C:\SAMPICsetups\mysetup.set'   # Uncomment to load a setup file
```

> **Note on the save directory:** this path is interpreted by the SAMPIC Windows GUI, so it must be a valid Windows path on that machine.

---

## Step 3 — Start the satellite

Open **Terminal 1** and navigate to the folder containing `sampic_satellite.py`.

```bash
python sampic_satellite.py --name One --group mylab
```

You should see output like:

```
[INFO] Satellite Sampic.One starting up ...
[INFO] Waiting for controller commands ...
```

The satellite is now in the **NEW** state, waiting for instructions from a controller.

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--name` | `SAMPIC` | Instance name (becomes `Sampic.<name>`) |
| `--group` | `default` | Must match the controller's `--group` |
| `--level` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING` |

---

## Step 4 — Start the controller

Open **Terminal 2**.

```bash
Controller -g mylab --config sampic_config.toml
```

The interactive IPython prompt appears:

```
mylab >
```

The `constellation` object and `cfg` (your loaded config) are available immediately.

---

## Step 5 — Verify the satellite is visible

```python
mylab > constellation.satellites
{'Sampic.One': SatelliteCommLink(name=One, class=Sampic)}
```

If the satellite does not appear, check that both processes use the same `--group` name.

---

## Step 6 — Initialize

This reads and stores all configuration parameters from the config file (IP, port, save directory, filenames, run limits, and setup file path). No network connection is made yet.

```python
mylab > constellation.initialize(cfg)
{'Sampic.One': SatelliteResponse(msg='Transition initialize is being initiated')}
```

Confirm the state:

```python
mylab > constellation.get_state()
{'Sampic.One': SatelliteResponse(msg='INIT', payload=32, ...)}
```

The satellite is now in **INIT**. Configuration is loaded and the satellite is ready to connect.

---

## Step 7 — Launch

Moves the satellite into the **ORBIT** (ready) state. This opens the TCP connection to the SAMPIC GUI and (if a `setup_file` is configured) sends `LOAD_SETUP` to load it.

```python
mylab > constellation.launch()
{'Sampic.One': SatelliteResponse(msg='Transition launch is being initiated')}
```

Check state:

```python
mylab > constellation.get_state()
{'Sampic.One': SatelliteResponse(msg='ORBIT', ...)}
```

---

## Step 8 — Start a run

This sends `START_RUN` to SAMPIC with all options from the config file.
SAMPIC begins acquiring and saving data to the configured directory on the Windows machine.

```python
mylab > constellation.start("1")   # argument is the run number
{'Sampic.One': SatelliteResponse(msg='Transition start is being initiated')}
```

Check state (should be **RUN**):

```python
mylab > constellation.get_state()
{'Sampic.One': SatelliteResponse(msg='RUN', ...)}
```

You will also see on the satellite terminal:

```
[INFO] SAMPIC acquisition started (run 1).
```

---

## Step 9 — Stop the run

```python
mylab > constellation.stop()
{'Sampic.One': SatelliteResponse(msg='Transition stop is being initiated')}
```

This sends `STOP_RUN` to the SAMPIC GUI. Acquisition ends and the data files are finalised on the Windows machine. The satellite returns to **ORBIT** state.

```python
mylab > constellation.get_state()
{'Sampic.One': SatelliteResponse(msg='ORBIT', ...)}
```

# To take another run, repeat Step 8 with an incremented run number:

```python
mylab > constellation.start("2")
```

---

## Step 10 — Land and shut down

When you are done with all runs:

```python
mylab > constellation.land()     # satellite → INIT state
mylab > constellation.reset()    # satellite → NEW state, TCP socket closed
```

Close the controller (does NOT affect the satellite):

```python
mylab > quit
```

Stop the satellite with `Ctrl+C` in Terminal 1.

---

## Full state machine summary

```
NEW  ──initialize──▶  INIT  ──launch──▶  ORBIT  ──start──▶  RUN
                                │                              │
                              reset                           stop
                                │                              │
                               NEW                          ORBIT
```

At any point, `constellation.get_state()` tells you where every satellite is.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Satellite not visible in controller | Group name mismatch | Ensure `--group` matches on both sides |
| `initialize` fails | SAMPIC GUI not running / wrong IP | Check the GUI is open and the TCP server is enabled; verify `sampic_ip` and `sampic_port` in the TOML |
| `START_RUN #EXECUTED KO` | No board connected in SAMPIC GUI, or acquisition already running | Check SAMPIC GUI status; call `stop()` first if a run is stuck |
| `STOP_RUN` no reply | Previous run already finished automatically (time/hit limit reached) | Safe to ignore; satellite will still transition to ORBIT |
| Data files not appearing | Wrong `save_directory` path | Path must be valid on the Windows machine running the GUI |

---

## Quick reference — all controller commands

```python
constellation.satellites             # list connected satellites
constellation.get_state()            # current FSM state of all satellites
constellation.initialize(cfg)        # NEW → INIT  (loads configuration)
constellation.launch()               # INIT → ORBIT (connects TCP, loads setup)
constellation.start("<run_number>")    # ORBIT → RUN  (sends START_RUN)
constellation.stop()                 # RUN → ORBIT  (sends STOP_RUN)
constellation.land()                 # ORBIT → INIT
constellation.reset()                # any → NEW    (closes socket)
constellation.Sampic.One.get_name()  # address a specific satellite directly
```
