#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd


# In[28]:


weather = pd.read_csv("local_weather.csv", index_col="DATE")


# In[29]:


weather


# In[30]:


weather.apply(pd.isnull).sum()/weather.shape[0]


# In[31]:


core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()


# In[32]:


core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]


# In[33]:


core_weather


# In[34]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[35]:


core_weather["precip"].value_counts()


# In[36]:


core_weather["precip"] = core_weather["precip"].fillna(0)


# In[37]:


core_weather["snow"] = core_weather["snow"].fillna(0)


# In[38]:


core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)


# In[39]:


core_weather[pd.isnull(core_weather["temp_min"])]


# In[40]:


core_weather = core_weather.fillna(method="ffill")


# In[41]:


core_weather.apply(pd.isnull).sum()


# In[42]:


core_weather.dtypes


# In[43]:


core_weather.index 


# In[44]:


core_weather.index = pd.to_datetime(core_weather.index)


# In[45]:


core_weather.index


# In[47]:


core_weather.index.month


# In[48]:


core_weather.apply(lambda x: (x==9999).sum())


# In[49]:


core_weather[["temp_max", "temp_min"]].plot()


# In[50]:


core_weather.index.year.value_counts().sort_index()


# In[51]:


core_weather["precip"].plot()


# In[54]:


core_weather["snow"].plot()


# In[55]:


core_weather["snow_depth"].plot()


# In[56]:


core_weather.groupby(core_weather.index.year).apply(lambda x: x["precip"].sum()).plot()


# In[59]:


core_weather.groupby(core_weather.index.year).sum()


# In[60]:


core_weather["target"] = core_weather.shift(-1)["temp_min"]


# In[61]:


core_weather


# In[63]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[64]:


core_weather


# In[65]:


from sklearn.linear_model import Ridge

reg = Ridge(alpha=.1)


# In[66]:


predictors = ["precip", "temp_max", "temp_min", "snow", "snow_depth"]


# In[67]:


train = core_weather.loc[:"2020-12-31"]
test = core_weather.loc["2021-01-01":]


# In[68]:


reg.fit(train[predictors], train["target"])


# In[69]:


predictions = reg.predict(test[predictors])


# In[70]:


from sklearn.metrics import mean_squared_error

mean_squared_error(test["target"], predictions)


# In[71]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[73]:


combined


# In[74]:


combined.plot()


# In[75]:


reg.coef_


# In[82]:


core_weather.corr()["target"]


# In[83]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[85]:


combined.sort_values("diff", ascending=False).head(15)


# In[ ]:




