import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn
from seaborn import stripplot,swarmplot,boxplot,violinplot,barplot,pointplot,boxenplot,countplot,distplot,kdeplot,rugplot,jointplot,pairplot,regplot,lmplot,residplot,relplot, heatmap

class Viz:

    def __init__(self,df,y=None):
        if y is None:
            self.y = None
        else:
            self.y = y

        self.df = df



    seaborn_styles = {
        'dark':'dark',
        'ticks':'ticks',
        'whitegrid':'whitegrid',
        'white':'white',

    }

    def select_charts(self,chartSelect,**kwargs):
        self.select_chart = chartSelect
        charts_dict = {
            'stripplot':stripplot,#Categorical scatterplots
            'swarmplot':swarmplot,#Categorical scatterplots
            'boxplot':boxplot,#Categorical distribution plots
            'violinplot':violinplot,#Categorical distribution plots
            'boxenplot':boxenplot,#Categorical distribution plots
            'pointplot':pointplot,#Categorical estimate plots
            'barplot':barplot,#Categorical estimate plots
            'countplot':countplot,#Categorical estimate plots
            'distplot':distplot,#univariate distributions
            'kdeplot':kdeplot,#univariate distributions
            'rugplot':rugplot,#univariate distributions
            'jointplot':jointplot,#bivarirate distributions
            'pairplot':pairplot, #pairwise relationship
            'regplot':regplot, #linear regression
            'lmplot':lmplot,#linear regression
            'residplot':residplot,#residual plot
            'relplot':relplot,#relating variables with scatter plots
            'qqplit':self.qqplot, #function for qq plot
            'heatmap':heatmap,
        }
        self.chartPlot =charts_dict[self.select_chart]

    def plot(self,x=None,y=None,**kwargs):
        if x is not None and y is not None:
            self.chartPlot(x=x,y=y,**kwargs)

    def qqplot(self,x,title,**kwargs):
        stats.probplot(x,**kwargs)
        plt.title(title)
        return plt


    

