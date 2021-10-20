from os import X_OK, path
import sys
from typing import Iterable
import numpy as np
sys.path.append('../')
sys.path.append('../code')
from bisearch import txt2map, SearchTree

def configFile():
    return {
        "outputPath":"result/18308133_liuxianbin_Astar.txt",
        "dataFile":"data/MazeData.txt",
        "toFile":False
    }

class MinHeap:
    def __init__(self, data:Iterable, key=None) -> None:
        self.heap = data
        self.key = key if key is not None else lambda x:x
        self.heap = sorted(self.heap, key=self.key)

    def __len__(self):
        return len(self.heap)
        
    def insert(self, ele):
        left, right = 0, self.__len__()-1
        if right < 0:
            self.heap.insert(0, ele)
            return
        val = self.key(ele)
        while abs(left - right)>=2:
            mid = int((left+right)/2)
            if val < self.key(self.heap[mid]): right = mid
            elif val > self.key(self.heap[mid]): left = mid
            else:
                self.heap.insert(mid, ele)
                return
        if val <= self.key(self.heap[left]): self.heap.insert(left, ele)
        elif val <= self.key(self.heap[right]): self.heap.insert(right, ele)
        else: self.heap.insert(right+1, ele)

    
    def pop(self):
        return self.heap.pop(0)

class Node:
    def __init__(self, pos, action=0, parent=None) -> None:
        self.pos = pos
        self.parent = parent
        self.action = action
        self.g = self.parent.g+1 if parent is not None else 0
        self.f = 0
        self.h = 0
    
    @property
    def x(self):return self.pos[0]
    @property
    def y(self):return self.pos[1]
    
    def calcf(self):
        self.f = self.g+self.h
    

class SearchTreePro(SearchTree):
    def __init__(self, txtfile: str) -> None:
        super().__init__(txtfile)
    
    def h(self, node:Node):
        return 0
        # return abs(node.x-self.e[0])+abs(node.y-self.e[1])

    def buildPath(self, last, State):
        Paths = ['E']
        curNode = State[last]
        while curNode.pos != self.s:
            Paths.append(self.a2char[curNode.action])
            curNode = curNode.parent
        Paths.append('S')
        self.Path = Paths[::-1]
        return self.compress()

    def AstarSearch(self, count=True):
        Cnt = 0
        root = Node(self.s)
        State = {self.s:root}
        boundry = MinHeap([root], key=lambda node:node.f)
        last = None
        found = 0
        while len(boundry)!= 0:
            if (found): break
            cur = boundry.pop()
            for son, action in self.neigh(cur.pos):
                son = tuple(son)
                action = int(action)
                pstate = self.map[son[0],son[1]]
                if pstate == 3:
                    last = son
                    found = 1
                    State[son]=Node(son, action, cur)
                    break
                if pstate == 0:
                    if tuple(son) in State:
                        # cut
                        continue
                    else:
                        Cnt += 1
                        newnode = Node(son, action, cur)
                        newnode.h = self.h(newnode)
                        newnode.calcf()
                        State[son]=newnode
                        # sort Boundry
                        boundry.insert(newnode)

        if count: print("共扩展%d个点"%(Cnt))
        if found: return self.buildPath(last, State)
        else: return []


def main():
    config = configFile()
    if config["toFile"]:
        with open(file=config["outputPath"], mode='w') as f:
            sys.stdout = f
    stp = SearchTreePro("data/MazeData.txt")
    print(stp.AstarSearch())

if __name__ == '__main__':
    main()