import xlearn as xl

# 设置Model
#model = xl.create_fm()
#model = xl.create_ffm()
model = xl.create_linear()

def train():
	model.setTrain("./attrition/train1.csv")
	# 设置参数
	param = {'task':'binary', 'lr':0.01, 'lambda':0.002, 'metric':'auc', 'epoch':100}

	# 模型训练
	#model.setTXTModel("./model.txt")
	#fm_model.cv(param)
	model.fit(param, "./model.out")

	# Use cross-validation
	#fm_model.cv(param)
def test():
	model.setTest("./attrition/test1.csv")
	# 通过Sigmoid转化到0-1之间
	#model.setSigmoid()
	# 通过setSign转化为0或1
	model.setSign()
	model.predict("./model.out", "./submit_xlearn_lr.txt")
	#model.predict("./model.out", "./submit_xlearn_fm.txt")

train()
test()
