# creacube_projet_s9

## Usage

### Execution 

* To launch tests: execute `pytest`
* To launch main function:
    * Create a python file
    * Add `from main import in_hand` at the beginning of the file
    * Prepare data according to ``Data`` section
    * Call `in_hand` function

### Data
`in_hand` functions required data preparation.
`in_hand` functions takes 2 arguments : 
* `positions` : which is a (nb_samples, 4, 3) numpy array that contains positions of the 4 cubes (x, y, z) coordinates for each sample.
* `is_connected` which is (nb_samples, 4, 6) boolean numpy array that contains if a face is connected to another for each face of each cube (Top, North, East, West, South, Bottom)

`in_hand` function return a which is (nb_samples, 4) array that contains for each sample if a cube is in hand or not for each cube. (True means in hand.)

## Files

* [main.py](./main.py) contains the main function to decide whether cubes are manipulated or not.
* [test_example.py](./test_example.py) contains test with "real" data.
* [test_fonct.py](./test_fonct.py) contains bigger tests with combinations of basic cases.
* [test_unit_fonct.py](./test_unit_fonct.py) contains basic tests of secondaries functions of [main.py](./main.py).
* [test_unit.py](./test_unit.py) contains basic tests of main function of [main.py](./main.py).
* [utils.py](./utils.py) contains function usefull for tests.
* [README.md](./README.md) 

## Algorithm

Our function is based on few rules to determine is the cube state changed after the last sample.
* Rule 1: If the cube is higher than 4 cubes, it is considered in hand.
* Rule 2: If the cube altitude is increasing, the cube is considered in hand. This some have some limitations:
    * If the structure of cube fall, one cube can go upward. This issue is corrected by rule 3, 4 and porpagation.
    * If the positions is not precise enough, the cube can be considered moving upward even if it don't move.
* Rule 3: If the cube is at the mat level and do not move, it is considered not in hand.
* Rule 4: If the cube had a constant alttude and move constantly for few samples, the cube is considered not in hand.
    * One issue of this rule is that it does not work with rotation movement on the mat and movement influenced by user.

After the application of theses rules, states of cubes are "propagate".
The "propagation" phase consist of the modification of state of cube connected to "not in hand" cube to "notin hand".
For example, if two cube are connected and one of these cube is considered "not in hand" after the propagation phase, both are considerer "not in hand".

Another precision on this algorithm :  If the user connect a cube to a cube structure "not in hand" and then deconnect it. This cube is considered "not in hand" for the samples of connection even if the user had it in hand. The main problem is that the data provided do not bring enough data to determine if the user hold the cube or not during the connection time.

## Tests to add
Our work is incomplete. Adding more complexe tests could be important to improve code reliability.
Here is some examples of relevant tests:
* A cube structure falls and/or splits in many parts.
* A cube structure turns on itself due to imbalance (for example).