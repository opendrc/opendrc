# This script is used to check the correctness of the results of our algorithm
# and the results of the klayout algorithm.

# parse arguments to get the path of our results and the path of klayout results
import argparse

import os


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"{self.x} {self.y}"

    def __eq__(self, o: object) -> bool:
        return abs(self.x-o.x)<=1 and abs(self.y-o.y)<=1

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class Edge:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"{self.start} {self.end}"

    def __eq__(self, o: object) -> bool:
        is_horizontal = self.start.y == self.end.y
        is_vertical = self.start.x == self.end.x

        if is_horizontal:
            start = min(self.start.x, self.end.x)
            end = max(self.start.x, self.end.x)
            ostart = min(o.start.x, o.end.x)
            oend = max(o.start.x, o.end.x)
            return abs(self.start.y-o.start.y)<=1 and not (start > oend or end < ostart)

        if is_vertical:
            start = min(self.start.y, self.end.y)
            end = max(self.start.y, self.end.y)
            ostart = min(o.start.y, o.end.y)
            oend = max(o.start.y, o.end.y)
            return abs(self.start.x-o.start.x)<=1 and not (start > oend or end < ostart)

        if self.start == o.start and self.end == o.end or self.end == o.start and self.start == o.end :
            return True


        return False

    def __hash__(self) -> int:
        return hash((self.start, self.end))


class Violation:
    def __init__(self, edge1, edge2):
        points = edge1.start, edge1.end, edge2.start, edge2.end

        self.edge1 = edge1
        self.edge2 = edge2
        points = sorted(points, key=lambda x: x.x)
        can1_edge1 = Edge(points[0], points[1])
        can1_edge2 = Edge(points[2], points[3])
        dist1 = points[-1].x - points[0].x
        can1_is_vert = can1_edge1.start.x == can1_edge1.end.x

        
        points = sorted(points, key=lambda x: x.y)
        can2_edge1 = Edge(points[0], points[1])
        can2_edge2 = Edge(points[2], points[3])
        dist2 = points[-1].y - points[0].y
        can2_is_hori = can2_edge1.start.y == can2_edge1.end.y
        if can1_is_vert or can2_is_hori:
            if can1_is_vert:
                self.edge1,self.edge2 = can1_edge1,can1_edge2
            if can2_is_hori:
                self.edge1,self.edge2 = can2_edge1,can2_edge2
            if can1_is_vert and can2_is_hori:
                if dist1 <dist2:
                    self.edge1,self.edge2 = can1_edge1,can1_edge2
                else:
                    self.edge1,self.edge2 = can2_edge1,can2_edge2
            


    def __repr__(self) -> str:
        return f"{self.edge1} {self.edge2}"

    def __eq__(self, o: object) -> bool:
        return (self.edge1 == o.edge1 and self.edge2 == o.edge2) or (
            self.edge1 == o.edge2 and self.edge2 == o.edge1
        )

    def __hash__(self) -> int:
        return hash((self.edge1, self.edge2))


def check(k_result_path, our_result_path):
    k_results = set()
    # read klayout results
    with open(k_result_path, "r") as f:
        for line in f:
            (
                edge1_x1,
                edge1_y1,
                edge1_x2,
                edge1_y2,
                edge2_x1,
                edge2_y1,
                edge2_x2,
                edge2_y2,
            ) = line.strip().split(" ")
            edge1 = Edge(
                Point(int(edge1_x1), int(edge1_y1)), Point(int(edge1_x2), int(edge1_y2))
            )
            edge2 = Edge(
                Point(int(edge2_x1), int(edge2_y1)), Point(int(edge2_x2), int(edge2_y2))
            )
            event = Violation(edge1, edge2)
            k_results.add(event)


    with open(our_result_path, "r") as f:
        not_found = 0

        for line in f:
            (
                edge1_x1,
                edge1_y1,
                edge1_x2,
                edge1_y2,
                edge2_x1,
                edge2_y1,
                edge2_x2,
                edge2_y2,
            ) = line.strip().split(" ")
            edge1 = Edge(
                Point(int(edge1_x1), int(edge1_y1)), Point(int(edge1_x2), int(edge1_y2))
            )
            edge2 = Edge(
                Point(int(edge2_x1), int(edge2_y1)), Point(int(edge2_x2), int(edge2_y2))
            )
            event = Violation(edge1, edge2)
            flag = False
            rm = None
            for k_event in k_results:
                if event == k_event:
                    flag = True
                    rm = k_event
                    break
            if flag:
                k_results.remove(rm)
            if not flag:
                print("Not found", event)
                not_found += 1
        print("in ", our_result_path, "Not found", not_found)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--our-result-path",
        type=str,
        default="./our_text",
        help="path to our text results",
    )
    parser.add_argument(
        "--k-result-path",
        type=str,
        default="./drc_text",
        help="path to klayout text results",
    )

    arg = parser.parse_args()

    kfiles = os.walk(arg.k_result_path)
    ourfiles = os.walk(arg.our_result_path)

    k_texts = set()
    our_texts = set()

    for path, _, file_list in kfiles:
        for file_name in file_list:
            k_texts.add(os.path.join(path, file_name))

    for path, _, file_list in ourfiles:
        for file_name in file_list:
            our_texts.add(os.path.join(path, file_name))

    for our_text in our_texts:
        text = our_text.strip().split("/")[-1]
        k_text = arg.k_result_path + "/" +text
        print(k_text)
        assert(k_text in k_texts)
        check(k_text,our_text);
    

