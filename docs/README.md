# Devloper Guide

## Design Philosophy

OpenDRC is an opensource design rule checker that supports GPU acceleration.
Initial project targets include 
- Support of GDSII
- Support of ASAP7 PDK
- Support of GPU acceleration

## Code Organization

### Logical organization
From top to bottom, OpenDRC consists of the following layers:
- Interface: IO, DB adaptor, function APIs
- Application: algorithm scheduling
- Algorithm: core algorithms, where GPU kernels are launched
- Infrastructure: core data structures, utilities, GPU libraries

In general, higher layers depend on the abstraction of lower layers, but are implementation-neutral.
Lower layers should not depend on higher layers.

### Physical (file) organization
OpenDRC basically follows the [canonical project structure](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1204r0.html#tests-unit) proposal, except that unittests are also placed in the `/test' folder.
Specifically, header files (`\*.hpp`) and source files (`\*.cpp`) are placed next to each other in the same folder (i.e., no `/include` and `/src` split!).

```
OpenDRC
+-- docs
+-- examples
+-- odrc
    +-- interface
        +-- gdsii
            +-- gdsii.cpp
            +-- gdsii.hpp
    +-- main.cpp
+-- tests
    +-- interface
        +-- gdsii
            +-- gdsii.test.cpp
    +-- main.cpp
+-- thirdparty
    +-- doctest
        +-- doctest.h
```

To include a header file, use
```c++
#include <odrc/interface/gdsii/gdsii.hpp>
```

## Test

## Building
OpenDRC uses CMake to automatically generate build system.
The standard way to build the project is as following:
```bash
mkdir build && cd build
cmake ..
make
```

## Development Workflow

### GitHub Flow
We adopt the GitHub flow (but not the git flow) for its simplicity, which better supports agile development.
Every change should be merged to the main repository by pull request.
In other words, a typical work flow consists of the following steps:

1. (do only once) Fork the repository, clone to local
2. Checkout to a new branch, e.g., `feat/gdsii`
3. Pull upstream updates
4. Code, code, code, ...
5. Commit to the forked repo
6. Open a pull request to the `main` branch

### Git commit message
A standard git commit message is as the following:
```
<type>(scope): <subject>
```

where type is one of the following:
- feat: add new feature
- fix: fix program bug(s)
- test: modify testing files
- perf: tune performance
- docs: update documentation
- chore: build system etc.
- style: only change style, not code logic
- refactor: code refactor
- revert: revert to previous version

scope is where the change takes place, and subject is a short summary of the commit.
The summary should use imperative verb form (i.e., present simple tense), avoid unnecessary capticalization, and not ended with punctuation.
In general the summary should fit in the following sentence:

> “If applied, my commit will (INSERT COMMIT MESSAGE TEXT)”

Whether to have more detailed description is up to the developer.
Here is a commit message example:
```
feat(iterface/gdsii): add parser for gdsii record header
```

## Coding Style
