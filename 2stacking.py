y_sub1, y_sub2, y_sub3, y_sub4, y_sub5 = y[:300], y[300: 600], y[600:900], y[900:1200],  y[1200:1459]

X_sub1, X_sub2, X_sub3, X_sub4, X_sub5 = X[:300], X[300: 600], X[600:900], X[900:1200], X[1200:1459]
allRidgeX = [X_sub1, X_sub2, X_sub3, X_sub4, X_sub5]
allRidgeY = [y_sub1, y_sub2, y_sub3, y_sub4, y_sub5]

ridgeTrain, ridgeTest = [], []
lassoTrain, lassoTest = [], []
elasticTrain, elasticTest = [], []
svrTrain, svrTest = [], []
gbrTrain, gbrTest = [], []
xgbTrain, xgbTest = [], []

for i in range(5): 
    ridge_model_full_data = ridge.fit(allRidgeX[i], allRidgeY[i])
    lasso_model_full_data = lasso.fit(allRidgeX[i], allRidgeY[i])
    elastic_model_full_data = elasticnet.fit(allRidgeX[i], allRidgeY[i])
    svr_model_full_data = svr.fit(allRidgeX[i], allRidgeY[i])
    gbr_model_full_data = svr.fit(allRidgeX[i], allRidgeY[i])
    xgb_model_full_data = xgboost.fit(allRidgeX[i], allRidgeY[i])
    xgbdata =  xgb_model_full_data.predict(allRidgeX[i])
    gbrdata =  svr_model_full_data.predict(allRidgeX[i])
    svrdata =  svr_model_full_data.predict(allRidgeX[i])
    elasticdata =  elastic_model_full_data.predict(allRidgeX[i])
    ridgedata = ridge_model_full_data.predict(allRidgeX[i])
    lassodata = lasso_model_full_data.predict(allRidgeX[i])
    for j in range(len(ridgedata)):
        ridgeTrain.append(ridgedata[j])
    for j in range(len(elasticdata)):
        elasticTrain.append(elasticdata[j])
    for j in range(len(lassodata)):
        lassoTrain.append(lassodata[j])
    for j in range(len(svrdata)):
        svrTrain.append(svrdata[j])
    for j in range(len(gbrdata)):
        gbrTrain.append(gbrdata[j])
    for j in range(len(xgbdata)):
        xgbTrain.append(xgbdata[j])


ridgedata1 = ridge_model_full_data.predict(X_sub)
lassodata1 = lasso_model_full_data.predict(X_sub)
elasticdata1 = elastic_model_full_data.predict(X_sub)
svrdata1 = svr_model_full_data.predict(X_sub)
gbrdata1 = gbr_model_full_data.predict(X_sub)
xgbdata1 = xgb_model_full_data.predict(X_sub)

for j in range(len(ridgedata1)):
        ridgeTest.append(ridgedata1[j])
for j in range(len(lassodata1)):
        lassoTest.append(lassodata1[j])
for j in range(len(elasticdata1)):
        elasticTest.append(elasticdata1[j])
for j in range(len(svrdata1)):
        svrTest.append(svrdata1[j])
for j in range(len(gbrdata1)):
        gbrTest.append(gbrdata1[j])
for j in range(len(xgbdata1)):
        xgbTest.append(xgbdata1[j])
        
tra1 = {'ridgeTrain': ridgeTrain,
       'lassoTrain': lassoTrain,
       'elasticTrain': elasticTrain,
       'svrTrain': svrTrain,
       'gbrTrain': gbrTrain,
       'xgbTrain':  xgbTrain}

tes1 = {'ridgeTrain': ridgeTest,
       'lassoTrain': lassoTest,
       'elasticTrain': elasticTest,
       'svrTrain': svrTest,
       'gbrTrain': gbrTest,
        'xgbTrain':  xgbTest}
trainFrame=pd.DataFrame(tra1)
testFrame=pd.DataFrame(tes1)

def cv_rmse1(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, trainFrame, testFrame, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

print(datetime.now(), 'stack_gen')
stack_gen_model = stack_gen.fit(np.array(trainFrame), np.array(y))

print(datetime.now(), 'elasticnet')
elastic_model_full_data = elasticnet.fit(trainFrame, y)

print(datetime.now(), '一范数lasso')
lasso_model_full_data = lasso.fit(trainFrame, y)

print(datetime.now(), '二范数')
ridge_model_full_data = ridge.fit(trainFrame, y)

print(datetime.now(), 'svr')
svr_model_full_data = svr.fit(trainFrame, y)

print(datetime.now(), 'GradientBoosting')
gbr_model_full_data = gbr.fit(trainFrame, y)

print(datetime.now(), 'xgboost')
xgb_model_full_data = xgboost.fit(trainFrame, y)


def blend_models_predict(X):
    return ((0.1111 * elastic_model_full_data.predict(X)) + \
            (0.0555 * lasso_model_full_data.predict(X)) + \
            (0.1111 * ridge_model_full_data.predict(X)) + \
            (0.1111 * svr_model_full_data.predict(X)) + \
            (0.1111 * gbr_model_full_data.predict(X)) + \
            (0.1667 * xgb_model_full_data.predict(X)) + \
            (0.3334 * stack_gen_model.predict(np.array(X))))


print('融合后的训练模型对原数据重构时的均方根对数误差RMSLE score on train data:')
print(rmsle(y, blend_models_predict(trainFrame)))

