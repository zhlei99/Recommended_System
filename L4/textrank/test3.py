#-*- encoding:utf-8 -*-
from textrank4zh import TextRank4Keyword, TextRank4Sentence

text = '土耳其国防部9日晚宣布，土军队已对叙利亚北部的库尔德武装展开军事行动。分析人士认为，此次行动并非土方突然之举，而是其在美国宣布从叙相关区域撤军后的必然选项。行动能否实现土所期待的清除叙库尔德武装、解决叙难民安置问题难以预料。但可以确定的是，此举势必会给当地民众带来灾难，会给叙局势乃至地区局势带来严重影响。土国防部表示，作为“和平之泉”军事行动的一部分，土军队已开始在叙利亚北部幼发拉底河以东地区发动地面进攻。土总统埃尔多安当日在社交媒体上发布了军事行动开始的消息，称“和平之泉”军事行动的目的是防止库尔德“恐怖分子”在土南部边境线上建立起一道“恐怖主义走廊”，并促使叙利亚难民重返家园。长期以来，土耳其都视叙库尔德武装为恐怖组织，欲将其清除。但美国却视这一武装为反恐作战合作伙伴，这让土耳其在采取行动时颇有顾忌。美土于8月曾同意在叙东北部建立一个“安全区”，隔离土边境和叙东北部的库尔德武装力量，但两国在“安全区”的范围和管理等方面迟迟无法达成一致。'

# 输出关键词，设置文本小写，窗口为2
tr4w = TextRank4Keyword()
tr4w.analyze(text=text, lower=True, window=2)
print('关键词：')
for item in tr4w.get_keywords(20, word_min_len=1):
    print(item.word, item.weight)

# 输出重要的句子
tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source = 'all_filters')
print('摘要：')
# 重要性较高的三个句子
for item in tr4s.get_key_sentences(num=3):
	# index是语句在文本中位置，weight表示权重
    print(item.index, item.weight, item.sentence)