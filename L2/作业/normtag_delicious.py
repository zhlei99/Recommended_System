
# 使用NormTagBased算法对Delicious数据进行推荐

import random
import math
import operator
 
class NormTagBased():
    
    # 构造函数
    def __init__(self,filename):
        self.filename=filename
        self.loadData()
        self.randomlySplitData(0.2)
        self.initStat()
        self.testRecommend()
        
    # 数据加载
    def loadData(self):
        print("使用NormTagBased算法，开始数据加载...")
        filename=self.filename
        # 保存了用户对item的tag
        self.records={}
        fi=open(filename)
        lineNum=0
        for line in fi:
            lineNum+=1
            if lineNum==1:
                continue
            uid,iid,tag,timestamp=line.split('\t')
            # 在数组中对应的下标-1
            uid=int(uid)-1
            iid=int(iid)-1
            tag=int(tag)-1
            self.records.setdefault(uid,{})
            self.records[uid].setdefault(iid,[])
            self.records[uid][iid].append(tag)
        fi.close()
        print("数据集大小为 %d." % (lineNum))
        print("设置tag的人数 %d." % (len(self.records)))
        #print(self.records)
        print("数据加载完成")
    
    # 将数据集拆分为训练集和测试集
    def randomlySplitData(self, ratio, seed=80):
        random.seed(seed)
        self.train=dict()
        self.test=dict()
        for u in self.records.keys():
            for i in self.records[u].keys():
                temp = random.random()
                # 对于用户u，将ratio比例的数据设置为测试集
                if temp<ratio:
                    self.test.setdefault(u,{})
                    self.test[u].setdefault(i,[])
                    for t in self.records[u][i]:
                        self.test[u][i].append(t)
                # 对于用户u，将1-ratio比例的数据设置为训练集
                else:
                    self.train.setdefault(u,{})
                    self.train[u].setdefault(i,[])
                    for t in self.records[u][i]:
                        self.train[u][i].append(t)
        print("训练集样本数 %d, 测试集样本数 %d" % (len(self.train), len(self.test)))
        #print('总的用户数 %d' %len(self.records.keys()))
    
    # 使用训练集，初始化user_tags, tag_items, user_items, item_tag
    def initStat(self):
        records=self.train
        self.user_tags=dict()
        self.tag_items=dict()
        self.user_items=dict()
        self.item_tags=dict()
        self.tag_users=dict()
        self.item_users=dict()
        for u,items in records.items():
            for i,tags in items.items():
                for tag in tags:
                    #print tag
                    # 用户和tag的关系
                    self._addValueToMat(self.user_tags,u,tag,1)
                    # tag和item的关系
                    self._addValueToMat(self.tag_items,tag,i,1)
                    # 用户和item的关系
                    self._addValueToMat(self.user_items,u,i,1)
                    # item和tag的关系
                    self._addValueToMat(self.item_tags,i,tag,1)
                    # tag和用户的关系
                    self._addValueToMat(self.tag_users,tag,u,1)
                    # items和用户的关系
                    self._addValueToMat(self.item_users,i,u,1)

        print("user_tags, tag_items, user_items, item_tags初始化完成.")
        print("user_tags大小 %d, tag_items大小 %d, user_items大小 %d" % (len(self.user_tags),len(self.tag_items),len(self.user_items)))
    
    # 设置矩阵 mat[index, item] = 1
    def _addValueToMat(self,mat,index,item,value=1):
        if index not in mat:
            mat.setdefault(index,{})
            mat[index].setdefault(item,value)
        else:
            if item not in mat[index]:
                mat[index][item] = value
            else:
                mat[index][item] += value
    
    # 使用测试集，计算准确率和召回率
    def precisionAndRecall(self,N):
        hit=0
        h_recall=0
        h_precision=0
        for user,items in self.test.items():
            if user not in self.train:
                continue
            # 获取Top-N推荐列表
            rank=self.recommend(user,N)
            for item,rui in rank:
                if item in items:
                    hit+=1
            h_recall+=len(items)
            h_precision+=N
        print('一共命中 %d 个, 一共推荐 %d 个, 用户设置tag总数 %d 个' %(hit, h_precision, h_recall))
        # 返回准确率 和 召回率
        return (hit/(h_precision*1.0)), (hit/(h_recall*1.0))
    
    # 对用户user推荐Top-N
    def recommend(self,user,N):
        recommend_items=dict()
        # 对Item进行打分，分数为所有的（用户对某标签使用的次数 wut, 乘以 商品被打上相同标签的次数 wti）之和
        tagged_items = self.user_items[user]
        # 遍历用户user打过的标签tags
        for tag, wut in self.user_tags[user].items():
            #print(self.user_tags[user].items())
            # 遍历tag打过的商品items
            for item, wti in self.tag_items[tag].items():
                if item in tagged_items:
                    continue
                #print('wut = %s, wti = %s' %(wut, wti))
                # 用户user打过的标签数 * 
                #print('用户user打过的标签数 %d, 商品被打上标签的次数 %d' %(len(self.user_tags[user].items()), len(self.item_tags[item].items())))
                #print('用户 %d, 商品 %d' %(user, item))
                # NormTagBased算法
                norm = len(self.tag_users[tag].items())
                # TagBased-TFIDF算法
                # norm = math.log(len(self.tag_users[tag].items()) + 1)
                #norm = math.log(len(self.tag_users[tag].items()) + 1) * math.log(len(self.item_tags[item].items()) + 1)
                if item not in recommend_items:
                    recommend_items[item] = wut * wti / norm
                else:
                    recommend_items[item] += wut * wti / norm
                #print(recommend_items[item])
        return sorted(recommend_items.items(), key=operator.itemgetter(1), reverse=True)[0:N]
    
    # 使用测试集，对推荐结果进行评估
    def testRecommend(self):
        print("推荐结果评估")
        #precision,recall=self.precisionAndRecall()
        print("%3s %20s %20s" % ('N',"精确率",'召回率'))
        for n in [5,10,20,40,60,80,100]:
        #for n in [100]:
            precision,recall=self.precisionAndRecall(n)
            print("%3d %19.3f%% %19.3f%%" % (n, precision * 100, recall * 100))
        
        
if __name__=='__main__':
    stb=NormTagBased("./user_taggedbookmarks-timestamps.dat")

