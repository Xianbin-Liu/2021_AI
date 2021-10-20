import sys
import numpy as np
sys.path.append('../')
sys.path.append('../code')

def configFile():
    return {
        "outputPath":"result/18308133_liuxianbin_bisearch.txt",
        "dataFile":"data/MazeData.txt"
    }

def txt2map(txtfile):
    amap = []
    s = (-1,-1)
    e = (-1,-1)
    with open(txtfile, "r") as f:
        amap = f.readlines()
    for i in range(len(amap)):
        indexs = amap[i].find('S')
        indexe = amap[i].find('E')
        amap[i] = list(amap[i].strip())
        if indexs != -1:
            s = (i, indexs)
            amap[i][indexs] = '2'
        if indexe != -1:
            e = (i, indexe)
            amap[i][indexe] = '3'
        amap[i] = list(map(int, amap[i]))
    return np.array(amap), s, e

class SearchTree:
    def __init__(self, txtfile:str) -> None:
        self.map, self.s, self.e = txt2map(txtfile)
        self.M, self.N = self.map.shape
        self.Path = []
        self.a2char = { # 1`4: left, right up down
            0 : "*",
            1 : "<-",
            2 : "->",
            3 : "^",
            4 : "v"
        }
        self.inv_a2char = { # 1`4: left, right up down
            0 : "*",
            1 : "->",
            2 : "<-",
            3 : "v",
            4 : "^"
        }
        self.a2pos = {
            1 : np.array([0,-1]),
            2 : np.array([0, 1]),
            3 : np.array([-1, 0]),
            4 : np.array([1, 0])
        }

    def biSearch(self, cnt=True):
        StartCnt, EndCnt = 0, 0
        if self.s == self.e:
            return [self.s]
        # state dict {pos : action}
        Ss = {self.s:0}
        Se = {self.e:0}
        # boundry
        fromStart = [self.s]
        fromEnd   = [self.e]

        found = 0
        meet = (-1,-1)
        while(len(fromStart)!= 0 or (len(fromEnd))!= 0):
            # start from Start set
            Ls = len(fromStart) # to calculate how many node should be visited
            while Ls != 0:
                Ls -= 1
                StartCnt += 1
                if (found): break
                cur = fromStart.pop(0)
                for son, action in self.neigh(cur):
                    state = self.map[son[0],son[1]]
                    if (state == 1) or (state == 2): continue
                    son = tuple(son)
                    if son in Se:
                        Ss[son] = int(action)
                        meet = son   # meet at this node
                        found = 1
                        break                   
                    elif son in Ss:
                        # cirle checking -- cut
                        continue
                    else:
                        # a new node
                        Ss[son] = int(action)
                        fromStart.append(son)
            if (found): break

            Le = len(fromEnd)
            while Le != 0:
                EndCnt += 1
                if (found): break
                Le -= 1
                cur = fromEnd.pop(0)
                for son, action in self.neigh(cur):
                    state = self.map[son[0],son[1]]
                    if state == 1 or state == 3: continue
                    son = tuple(son)
                    if son in Ss:
                        found = 1
                        Se[son] = int(action)
                        meet = son
                        break                   
                    elif son in Se:
                        # cirle checking -- cut
                        continue
                    else:
                        # a new node
                        Se[son] = int(action)
                        fromEnd.append(son)    
            if (found): break

        if cnt: print("从起点处共扩展：%d个点；从终点共扩展%d个点"%(StartCnt, EndCnt))

        if found:
            return self.buildPath(meet, Ss, Se)
        else: return []

    def buildPath(self, meet, Ss, Se):
        Paths, Pathe = [], []
        endS = endE = meet
        while endS != self.s:
            action = Ss[endS]
            Paths.append(self.a2char[action])
            endS = tuple(np.array(endS)-self.a2pos[action])
        Paths.append('*')
        while endE != self.e:
            action = Se[endE]
            Pathe.append(self.inv_a2char[action])
            endE = tuple(np.array(endE)-self.a2pos[action])
        Pathe.append('*')
        self.Path = Paths[::-1]+Pathe
        self.compress()
        return self.Path

    def compress(self):
        Paths= self.Path
        compressPath = []
        record, cnt = Paths[0], 1
        for cur in Paths[1:]:
            if cur == record: cnt += 1
            else:
                if cnt > 1: compressPath.append((record, cnt))
                else: compressPath.append(record)
                record, cnt = cur, 1
        self.Path = compressPath
        return compressPath

    def neigh(self, pos):
        # 1,2,3,4 means left, right, up, down
        mask = np.array([[0,-1],[0,1],[-1,0],[1,0]])
        action = np.array([[1],[2],[3],[4]])
        sons = mask+np.array(pos)
        index= ((sons >= 0) & (sons < [self.M,self.N])).all(axis=1)
        sons, action = sons[index], action[index]
        for i in range(len(sons)): yield sons[i],action[i]


def main():
    config = configFile()
    with open(file=config["outputPath"], mode='w') as f:
        sys.stdout = f
        st1 = SearchTree(config["dataFile"])
        print(st1.biSearch())

if __name__ == '__main__':
    main()