# OpenDRC User Manual

**OpenDRC** is an efficient Open-Source design rule checking (DRC) engine with hierarchical GPU acceleration.

Four steps to use **OpenDRC**.
1. initialize the engine with rule information and mode.
2. assign the gds database.
3. for each rule, do design rule checking.
4. output the violations.

There is an example:

```c++
auto db = odrc::gdsii::read(filename);
auto e = odrc::core::engine();
e.add_rules({
    e.layer(19).width().greater_than(18),
    e.layer(19).spacing().greater_than(18),
    e.layer(19).with_layer(21).enclosure().greater_than(2),
    e.layer(19).area().greater_than(504),
});
e.set_mode(mode::sequential);
e.check(db);
```

___
## Core Data Structures

### Database

**OpenDRC** has a GDSII parser to read GDSII files and store hierarchy data in the database. The minimum function unit is `cell` which contains multiple `polygon` and instantiates other `cell`s.

You can use the following code to parse a GDSII file.

```c++
auto db = odrc::gdsii::read(filename);
```

### Interval Tree

The interval tree is a data structure which queries all intervals overlapping with the given interval (or point) efficiently. In OpenDRC, we use it and the sweepline algorithm to get all overlapping `cells`.

```c++
template <typename T, typename V>
class interval_tree {
 public:
  using Intvl = interval<T, V>;
  using Node  = node<T, V>;
  using Ovlp  = std::vector<std::pair<V, V>>;
  void insert(const Intvl& intvl)
  void remove(const Intvl& intvl) 
  void get_intervals_pairs(const Intvl& intvl, Ovlp& ovlps)
 private:
  std::vector<Node> nodes;
};
```

An interval tree is a binary search tree (BST) that stores an `interval` $I$ in the highest `node` satisfying $u \in I$, where $u$ is the key of this node.
`insert` and `remove` manage the intervals of tree dynamically with time complexity of `O(log(n))`. `get_intervals_pairs` saves the overlapping cell information pairs in `ovlps`.

### Engine

The capabilities of the DRC feature include:

1. Mode option.
2. Rule addition.
3. Function scheduler (todo).

```c++
class engine {
 public:
  ...
  mode                   check_mode = mode::sequential;
  std::vector<std::pair<int, std::pair<int, int>>> vlt_paires;

  void add_rules(std::vector<int> value) 
  void set_mode(mode md) { check_mode = md; };
  void check(odrc::core::database& db) 

 private:
  ...
};
```

The `check` function will execute swtich case to implement the corresponding check function. If you need add a new design rule, you should add a new case in `engine` and add its check function in `algorithm`.  

## Algorithm

### Preprcocess

#### Layout Partition

```c++
std::vector<std::vector<int>> layout_partition( odrc::core::database& db,
                                                std::vector<int> layers,
                                                int threshold = 18)
```


The input of interval merging is a set $S$ of intervals to be merged and the output is non-overlapping intervals covering the domain of $S$. But in actual implementation, we pass the gds `database`, `layers` information and rule `threshold` to the function and it will return `cell_refs` ids of each row.


The overall procedure is as follows:

1. extract `cell_refs` on the specific layers from the gds database
2. enlarge the minimum bounding box of each `cell_ref` by the rule threshold and discretize the **y** coordinates of the enlarged boxes
3. view all enlarged boxes as intervals to be merged along the **y** axis and merge them
4. cluster the `cell_refs` in the same interval after merging and return them

### Check Algorithms

#### Width Check

Width check is an intra-cell check. It ensures that the distance between interior side of edges of a specific layer is not less than the threshold.

You can use the following code to implement width check.

```c++
void width_check_seq(odrc::core::database&         db,
                     int                           layer,
                     int                           threshold,
                     std::vector<core::violation>& vios) 
```

`db` refers database from user GDSII file, `layer` and `threshold` are determined by design rule. `vio` stores the violation pair information. Width check is hierarchical. This function checks once for the same cell and gets violation pairs in the form of virtual coordinates. Then the real coordinates could be calculated by reference coordinates.

#### Space Check

Space check ensures that the distance between the exterior side of edges of one or two layers is not less than the threshold.

You can use the following code to implement space check.

```c++
void space_check_seq(odrc::core::database&         db,
                     std::vector<int>              layers,
                     std::vector<int>              without_layer,
                     int                           threshold,
                     core::rule_type               ruletype,
                     std::vector<core::violation>& vios)
```

`db` refers database from the user GDSII file. `layer`, `without_layer`,`ruletype` and `threshold` are utilized to select right edge pairs. `vio` stores the violation pair information. 

There are two types of space checks, `inter-cell` and `intra-cell`.
For inter-cell space check, this function would partition the layout into many rows. We expand the minimum boundary rectangle (MBR) by threshold and utilize the interval tree to get all overlapping cell pairs in every row. Then we just need to check all cell pairs to get violations. The intra-cell space check is also hierarchical like width check.

#### Enclosure Check

Enclosure check ensures the distance between the exterior side of one layer's edge and the interior side of another layer's edge, where one of the layer geometry lies completely within another layer geometry is not less than threshold.

You can use the following code to implement enclosure check.

```c++
void enclosure_check_seq(odrc::core::database&         db,
                         const std::vector<int>&       layers,
                         const std::vector<int>&       without_layer,
                         const int                     threshold,
                         const core::rule_type         ruletype,
                         std::vector<core::violation>& vios) 
```

`db` refers database from the user GDSII file. `layers`, `without_layer`,`ruletype` and `threshold` are utilized to select right edge pairs. `vio` stores the violation pair information. `layers` should contain two elements: the first is the metal layer and the second is the via layer.

This function would partition the layout into many rows. We expand the MBR of via cell by threshold and utilize the interval tree to get all overlapping cell pairs in every row. Then we just need to check all cell pairs to get violations.

#### Area Check

Area check ensures the polygon area is not less than threshold.

You can use the following code to implement area check.

```c++
void area_check_seq(const odrc::core::database&   db,
                    int                           layer,
                    int                           threshold,
                    std::vector<core::violation>& vios)
```

`db` refers database from user GDSII file, `layer` and `threshold` are determined by design rule. `vio` stores the violation pair information. Area check is also hierarchical like width check.

## Check Result with Klayout

After getting checking results, we use Klayout to generate the true results of design rule checking. We transform the output gds files of **Klayout** DRC scripts into TXT files.
Then we compare the TXT files with the output of our program. 

The overall comparing idea is as follows:

1. read the violation information from the klayout TXT file
2. for each violation in our output, check whether it is in the **Klayout** TXT file. If it is, delete it from the klayout TXT file and it is a true violation. If it is not, it is a false positive.

