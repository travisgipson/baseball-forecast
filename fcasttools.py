# lib import
import pandas as pd
import numpy as np
import pybaseball.datahelpers as pbbdata
import pybaseball.analysis.projections.marcels.age_adjustment as pbb_age_adj
import statsmodels.api as sm
import xgboost as xgb
from tqdm.notebook import tqdm_notebook

class CoreTeamETL:
    
    def __init__(self,df_tm,dt_filter):
        
        # init data and params
        self.df_tm = df_tm
        self.dt_filter = dt_filter
        
    def pre_process_tm_data(self,df,dt_filter):
        
        # sort values
        df = df.sort_values(["teamID","yearID"])
        
        # drop cols
        df = df.drop(["franchID","Rank","W","L","G","Ghome",
                      "park","attendance","teamIDBR","teamIDretro"],axis=1).copy()
        
        # rename cols
        df = df.rename({"name":"teamName"},axis=1)
        
        # filter by date
        if dt_filter is not None:
            df = df.query(f"yearID >= {dt_filter}")
            
        return df
    
    def augment_primary_data(self,df):
        
        # adjust playoff appearance cols
        df["DivWin"] = df["DivWin"].str.replace("Y","1").replace("N","0").astype(int)
        df["WCWin"] = df["WCWin"].str.replace("Y","1").replace("N","0").astype(int)
        df["LgWin"] = df["LgWin"].str.replace("Y","1").replace("N","0").astype(int)
        df["WSWin"] = df["WSWin"].str.replace("Y","1").replace("N","0").astype(int)
        df["PSWin"] = (df[["DivWin","WCWin","LgWin","WSWin"]].sum(1) / df[["DivWin","WCWin","LgWin","WSWin"]].sum(1)).fillna(0).astype(int)
        df["Lg+Win"] = (df[["LgWin","WSWin"]].sum(1) / df[["LgWin","WSWin"]].sum(1)).fillna(0).astype(int)
        df["Div+Win"] = (df[["DivWin","LgWin","WSWin"]].sum(1) / df[["DivWin","LgWin","WSWin"]].sum(1)).fillna(0).astype(int)
        
        return df
    
    def run_etl(self):
        
        # pre-process data
        self.df_primary = self.pre_process_tm_data(df=self.df_tm,dt_filter=self.dt_filter)
        
        # augment primary dataset
        self.df_primary = self.augment_primary_data(df=self.df_primary)
        
        # re-arrange cols
        self.cols_team = ["teamID","teamName","lgID","divID","yearID"]
        self.cols_y = ["WCWin","DivWin","LgWin","WSWin","PSWin","Lg+Win","Div+Win"]
        self.cols_x = numcols = [c for c in self.df_primary.columns if c not in self.cols_team + self.cols_y]
        self.df_primary = self.df_primary[self.cols_team + self.cols_y + self.cols_x]
        
        return self


class BattingETL:
    
    def __init__(self,df_batting,df_fielding,df_players,dt_filter,pstn_filter,season_filter):
        
        # init data and parameters
        self.df_batting = df_batting
        self.df_fielding = df_fielding
        self.df_players = df_players
        self.dt_filter = dt_filter
        self.pstn_filter = pstn_filter
        self.season_filter = season_filter
    
    def pre_process_player_data(self,df):
        
        # drop players where debut is null
        df = df[~df["debut"].isna()].copy()
        
        # adjust debut dtype
        df["debut"] = pd.to_datetime(df["debut"])

        # create player name column
        df["playerName"] = df["nameFirst"] + df["nameLast"]
                        
        # filter columns
        df = df[["playerID","playerName","birthYear","debut","weight","height","bats","throws"]]
        
        return df

    def pre_process_batting_data(self,df,dt_filter):
        
        # sort values
        df = df.sort_values(["playerID","yearID","stint"])
        
        # filter dates
        if dt_filter is not None:
            df = df.query(f"yearID >= {dt_filter}")
        
        # augment batting data
        df = pbbdata.postprocessing.augment_lahman_batting(df)
                
        return df
    
    def group_batter_data(self,df):
        
        # duplicate league and team ids for aggregation
        df["teamID2"] = df["teamID"].copy()
        df["lgID2"] = df["lgID"].copy()
        
        # list numeric columns to aggregate
        num_cols = df.drop(["playerID","yearID","teamID","lgID","teamID2","lgID2","stint"],axis=1).columns.tolist()
        
        # initialize aggregation dict
        d = {"teamID":"first",
             "teamID2":"last",
             "lgID":"first",
             "lgID2":"last"}
        
        # update dict for each numeric column
        [d.update({x:"sum"}) for x in num_cols]
        
        # group by player, season
        df = df.groupby(["playerID","yearID"]).agg(d).reset_index()
        
        # rename columns
        df = df.rename({"teamID":"tmFirst",
                        "teamID2":"tmLast",
                        "lgID":"lgFirst",
                        "lgID2":"lgLast"},axis=1)
        
        return df

    def pre_process_fielding_data(self,df):
        
        # utilize pybaseball's method
        df = pbbdata.transform.get_primary_position(df)
        
        return df
        
    def merge_primary_data(self,df_batting,df_fielding,df_players):
        
        # merge data
        df = df_players.merge(df_batting,on=["playerID"],how="right").merge(df_fielding,on=["playerID","yearID"],how="left")
                      
        return df
    
    def augment_primary_data(self,df,pstn_filter,season_filter):
        
        # filter primaryPos types
        if pstn_filter is not None:
            df = df[~df["primaryPos"].isin(pstn_filter)].copy()
            
        # fill nan primaryPos types
        df["primaryPos"] = df.groupby("playerID")["primaryPos"].fillna(method="ffill")

        # create player ages
        df["age"] = df["yearID"] - df["birthYear"]
        df["ageAdj"] = df["age"].apply(pbb_age_adj.age_adjustment)
        
        # create season counter
        df["seasonNo"] = df.assign(seasonNo = 1).groupby("playerID")["seasonNo"].cumsum().astype(str)
        df["seasonGrp"] = np.where(df["seasonNo"].astype(int) < 3, "Rookie",
                                   np.where((df["seasonNo"].astype(int) >= 3) & (df["seasonNo"].astype(int) < 8), "Prime",
                                   np.where(df["seasonNo"].astype(int) >= 8, "Veteran","Other")))
                
        # filter out problematic seasons
        if season_filter is not None:
            df = df[~df["yearID"].isin(season_filter)].copy()
                
        # drop cols
        df = df.drop(["birthYear"],axis=1)
        
        return df

    def run_etl(self):
        
        # process player features
        self.df_players = self.pre_process_player_data(df=self.df_players)

        # process batting features
        self.df_batting = self.pre_process_batting_data(df=self.df_batting,dt_filter=self.dt_filter)
        
        # group batting features by player, season
        self.df_batting_gb = self.group_batter_data(df=self.df_batting)
        
        # process fielding features
        self.df_fielding = self.pre_process_fielding_data(df=self.df_fielding)
        
        # merge all primary datasets
        self.df_primary = self.merge_primary_data(df_batting=self.df_batting_gb,
                                                  df_players=self.df_players,
                                                  df_fielding=self.df_fielding)
        
        # augment the primary dataset
        self.df_primary = self.augment_primary_data(df=self.df_primary,
                                                    pstn_filter=self.pstn_filter,
                                                    season_filter=self.season_filter)
        
        # set final player columns
        self.cols_player = ["playerID","playerName","yearID","tmFirst","tmLast",
                            "lgFirst","lgLast","debut","seasonNo","seasonGrp","age","ageAdj",
                            "primaryPos","weight","height","bats","throws"]
        
        # set final batter columns
        self.cols_batter = [c for c in self.df_primary.columns if c not in self.cols_player]
        
        # re-arrange columns and do a final filter
        self.df_primary = self.df_primary[self.cols_player + self.cols_batter]

        return self

    
class PitchingETL:
    
    def __init__(self,df_pitching,df_players,dt_filter,pstn_filter,season_filter):
        
        # init data and parameters
        self.df_pitching = df_pitching
        self.df_players = df_players
        self.dt_filter = dt_filter
        self.pstn_filter = pstn_filter
        self.season_filter = season_filter
    
    def pre_process_player_data(self,df):
        
        # drop players where debut is null
        df = df[~df["debut"].isna()].copy()
        
        # adjust debut dtype
        df["debut"] = pd.to_datetime(df["debut"])

        # create player name column
        df["playerName"] = df["nameFirst"] + df["nameLast"]
                        
        # filter columns
        df = df[["playerID","playerName","birthYear","debut","weight","height","bats","throws"]]
        
        return df

    def pre_process_pitching_data(self,df,dt_filter):
        
        # sort values
        df = df.sort_values(["playerID","yearID","stint"])
        
        # filter dates
        if dt_filter is not None:
            df = df.query(f"yearID >= {dt_filter}")
        
        # augment pitching data
        df = pbbdata.postprocessing.augment_lahman_pitching(df)
                
        return df
    
    def group_pitcher_data(self,df):
        
        # duplicate league and team ids for aggregation
        df["teamID2"] = df["teamID"].copy()
        df["lgID2"] = df["lgID"].copy()
        
        # list numeric columns to aggregate
        num_cols = df.drop(["playerID","yearID","teamID","lgID","teamID2","lgID2","stint"],axis=1).columns.tolist()
        
        # initialize aggregation dict
        d = {"teamID":"first",
             "teamID2":"last",
             "lgID":"first",
             "lgID2":"last"}
        
        # update dict for each numeric column
        [d.update({x:"sum"}) for x in num_cols]
        
        # group by player, season
        df = df.groupby(["playerID","yearID"]).agg(d).reset_index()
        
        # rename columns
        df = df.rename({"teamID":"tmFirst",
                        "teamID2":"tmLast",
                        "lgID":"lgFirst",
                        "lgID2":"lgLast"},axis=1)
        
        return df
        
    def merge_primary_data(self,df_pitching,df_players):
        
        # merge data
        df = df_players.merge(df_pitching,on=["playerID"],how="right")
                      
        return df
    
    def augment_primary_data(self,df,pstn_filter,season_filter):
        
        # filter primaryPos types
        if pstn_filter is not None:
            df = df[~df["primaryPos"].isin(pstn_filter)].copy()
            
        # set primaryPos as Pitcher (P)
        df["primaryPos"] = "P"

        # create player ages
        df["age"] = df["yearID"] - df["birthYear"]
        df["ageAdj"] = df["age"].apply(pbb_age_adj.age_adjustment)
        
        # create season counter
        df["seasonNo"] = df.assign(seasonNo = 1).groupby("playerID")["seasonNo"].cumsum().astype(str)
                
        # filter out problematic seasons
        if season_filter is not None:
            df = df[~df["yearID"].isin(season_filter)].copy()
            
        # rename columns
        # df = df.rename({"R":"RA","H":"HA"},axis=1)
        
        # drop cols
        df = df.drop(["birthYear"],axis=1)
        
        return df
    
    def run_etl(self):
        
        # process player features
        self.df_players = self.pre_process_player_data(df=self.df_players)

        # process pitching features
        self.df_pitching = self.pre_process_pitching_data(df=self.df_pitching,dt_filter=self.dt_filter)
        
        # group pitching features by player, season
        self.df_pitching_gb = self.group_pitcher_data(df=self.df_pitching)
                
        # merge all primary datasets
        self.df_primary = self.merge_primary_data(df_pitching=self.df_pitching_gb,
                                                  df_players=self.df_players)
        
        # augment the primary dataset
        self.df_primary = self.augment_primary_data(df=self.df_primary,
                                                    pstn_filter=self.pstn_filter,
                                                    season_filter=self.season_filter)
        
        # set final player columns
        self.cols_player = ["playerID","playerName","yearID","tmFirst","tmLast",
                            "lgFirst","lgLast","debut","seasonNo","age","ageAdj",
                            "primaryPos","weight","height","bats","throws"]
        
        # set final batter columns
        self.cols_pitcher = [c for c in self.df_primary.columns if c not in self.cols_player]
        
        # re-arrange columns and do a final filter
        self.df_primary = self.df_primary[self.cols_player + self.cols_pitcher]

        return self


class PlayerForecastAR:
    
    def __init__(self,model_name,data_class,groupers,ar_type,player_model,group_model,nlags,lookback):
        
        # init data and params
        self.model_name = model_name
        self.data_class = data_class
        self.groupers = groupers
        self.ar_type = ar_type
        self.player_model = player_model
        self.group_model = group_model
        self.nlags = nlags
        self.lookback = lookback
                
    def get_lagged_data(self,df,stat_cols,nlags):
        
        # iterate over each stat col
        for col in stat_cols:
            
            # iterate over each lag
            for lag in range(1,nlags+1):
                
                # create lag series
                df[f"{col}_lag{lag}"] = df.groupby("playerID")[col].shift(lag)
                
        return df
    
    def get_updated_player_data(self,df,stat_cols):
        
        # update player data for out of sample forecasts
        df["yearID"] = df["yearID"] + 1
        df["seasonNo"] = df["seasonNo"] + 1
        df["age"] = df["age"] + 1
        df["ageAdj"] = df["age"].apply(pbb_age_adj.age_adjustment)
        df["isOpRost"] = np.nan
        df["tmFirst"] = "-"
        df["lgFirst"] = "-"
        
        return df
        
    def get_train_test_split(self,df,stat_cols,season,nlags,lookback):
        
        # locate the max available season data
        max_season = df["yearID"].max()
        
        # set the out of sample (test) date range
        test_start = season
        
        # set the train date ranges
        train_end = test_start - 1
        train_start = train_end - lookback + 1
        
        # set the necessary columns for model
        model_cols = self.data_class.cols_player + stat_cols
        
        # filter df columns
        df_filt = df[model_cols].copy()
        
        # control for forecasts outside of historical data range
        if test_start > max_season:
            
            # raise error if > 1 yr diff
            if test_start - max_season > 1:
                
                raise Exception("Error: Cannot forecast > 1 year after hirstorical data ends.")
            
            # normal case
            else:
                
                # create temp data with a copy of last available data
                temp = df_filt.query(f"yearID == {max_season}").copy()
                
                # update the temp player data (add a year to various cols)
                temp = self.get_updated_player_data(df=temp,stat_cols=stat_cols)
                
                # append
                df_filt = pd.concat([df_filt,temp],axis=0)
        
        # get lagged data
        df_filt = self.get_lagged_data(df=df_filt,stat_cols=stat_cols,nlags=nlags)
        
        # create indicator for not meeting the minimum season requirement
        df_filt["isNewPlayer"] = (df_filt["seasonNo"].copy().astype(int) <= nlags) * 1
        
        # add the previous year column for reference
        df_filt["yearIDPrev"] = df_filt["yearID"] - 1

        # split into train / test samples
        df_train = df_filt.query(f"yearID >= {train_start} and yearID <= {train_end}").copy()
        df_test = df_filt.query(f"yearID == {test_start}").copy()
                        
        return df_train, df_test
    
    def fit_group_model(self,df,yvar,groupers,group_model,group_lookback):
        
        # group by groupers + year
        dfg = df.groupby(groupers + ["yearID"])[[yvar]].agg(group_model).reset_index()
                
        return dfg.set_index("yearID").groupby(groupers).rolling(group_lookback).agg(group_model).reset_index()
    
    def train_model(self,df_train,yvar,xvars,groupers,player_model,group_model,group_lookback):
        
        # subset training data
        df_train_with_model = df_train.query(f"isNewPlayer == 0").dropna().copy()
        df_train_with_group = df_train.query(f"isNewPlayer == 1").copy()
        
        # set y for player models
        y = df_train_with_model[yvar].copy()
        
        # set X for player models
        X = pd.get_dummies(df_train_with_model[xvars].copy())
        
        # fit player model
        player_model = player_model.fit(y=y,X=X)
        
        # fit group model
        group_model = self.fit_group_model(df=df_train_with_group,
                                           yvar=yvar,
                                           groupers=groupers,
                                           group_model=group_model,
                                           group_lookback=group_lookback)
        
        return player_model, group_model

    def forecast(self,df_test,yvar,xvars,player_model,group_model):
        
        # copy df
        df_test = df_test.copy()
                
        # get player prediction df        
        X = pd.get_dummies(df_test[xvars].copy())
        
        # no nans / with nans
        Xnn = X[~X.isnull().any(axis=1)].copy()
        Xwn = X[X.isnull().any(axis=1)].copy()
        
        # get player forecasts
        Xnn[f"{yvar}"] = player_model.predict(Xnn)
        
        # join to test data
        df_test[f"{yvar}"] = pd.concat([Xnn,Xwn])[f"{yvar}"]
        
        # merge group forecast
        df_test = df_test.reset_index().merge(group_model,
                                              how="left",
                                              left_on=group_model.rename({"yearID":"yearIDPrev"},axis=1).columns.drop(yvar).tolist(),
                                              right_on=group_model.columns.drop(yvar).tolist(),
                                              suffixes=["","grp"]).set_index("index")
        
        # fill player forecast nan's
        df_test[f"{yvar}"] = df_test[f"{yvar}"].fillna(value=df_test[f"{yvar}grp"])
        
        return df_test[[f"{yvar}"]]
    
    def fetch_ar_vars(self,y,stats):
        
        # get base case ar xvars
        if self.ar_type == "ar":
            xvars = [f"{y}_lag{l}" for l in range(1,self.nlags+1)]
        
        # if var, get extra xvars
        if self.ar_type == "var":
            xvars = []
            for s in stats:
                xvars += [f"{s}_lag{l}" for l in range(1,self.nlags+1)]
       
        # get base case ar xvars
        if self.ar_type == "ar-groups":
            xvars = [f"{y}_lag{l}" for l in range(1,self.nlags+1)]
            xvars += self.groupers
        
        # if var, get extra xvars
        if self.ar_type == "var-groups":
            xvars = []
            for s in stats:
                xvars += [f"{s}_lag{l}" for l in range(1,self.nlags+1)]
            xvars += self.groupers
        
        return xvars
    
    def projections(self,season,yvar,xvars):
        
        # check stat type
        if type(yvar) is str:
            yvar = [yvar]
        
        # get test / train splits
        df_train, df_test = self.get_train_test_split(df=self.data_class.df_primary,
                                                      stat_cols=yvar,
                                                      season=season,
                                                      nlags=self.nlags,
                                                      lookback=self.lookback)
        
        # save outputs here
        df_forecasts = pd.DataFrame(index=df_test.index)
        player_models = dict()
        group_models = dict()
        
        # iterate over each stat to forecast
        for y in yvar:
            
            # set xvars
            xregs = self.fetch_ar_vars(y=y,stats=yvar) + xvars
                        
            # train model
            player_models[y], group_models[y] = self.train_model(df_train=df_train,
                                                                 yvar=y,
                                                                 xvars=xregs,
                                                                 groupers=self.groupers,
                                                                 player_model=self.player_model,
                                                                 group_model=self.group_model,
                                                                 group_lookback=self.nlags)
            
            # get forecast
            df_forecasts = df_forecasts.join(self.forecast(df_test=df_test,
                                                           yvar=y,
                                                           xvars=xregs,
                                                           player_model=player_models[y],
                                                           group_model=group_models[y]))
            
        # merge player + forecast data
        return df_test[["playerID","yearID"]].join(df_forecasts.copy()).set_index(["playerID","yearID"])


class TeamForecast:
    
    def __init__(self,model_name,train_corpus,test_corpus,data_class,model,lookback):
        
        # initialize data / params
        self.model_name = model_name
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
        self.data_class = data_class
        self.model = model
        self.lookback = lookback
    
    def get_train_test_split(self,train_corpus,test_corpus,model_cols,season,lookback):
        
        # locate the max available season data
        max_season = train_corpus["yearID"].max()
        
        # set the out of sample (test) date range
        test_start = season
        
        # set the train date ranges
        train_end = test_start - 1
        train_start = train_end - lookback + 1
                
        # filter columns
        train_filt = train_corpus[["teamID","yearID"] + model_cols].copy()
        test_filt = test_corpus[["teamID","yearID"] + model_cols].copy()
        
        # control for forecasts outside of historical data range (need to fix this)
        if test_start > max_season:
            
            # raise error if >= 1 yr diff
            if test_start - max_season >= 1:
                
                raise Exception("Error: Must insert prediction df if season outside of historical range.")
                
        # split into train / test samples
        df_train = train_filt.query(f"yearID >= {train_start} and yearID <= {train_end}").copy()
        df_test = test_filt.query(f"yearID == {test_start}").copy()
                        
        return df_train, df_test
        
    def forecast(self,df_test,yvar,xvars,model,pred_type):
        
        # copy df
        df_test = df_test.copy()
                        
        # get player forecast
        if pred_type == "class":
            df_test[f"{yvar}"] = model.predict(df_test[xvars])
        if pred_type == "prob":
            df_test[f"{yvar}"] = (pd.DataFrame(model.predict_proba(df_test[xvars]))[1].values).round(4)
        
        return df_test[[f"{yvar}"]]
    
    def projections(self,season,yvar,xvars,pred_type):
        
        # check yvar type
        if type(yvar) is str:
            yvar = [yvar]
            
        # get xvars
        if xvars == "all":
            xvars = self.data_class.cols_x
   
        # get test / train splits
        df_train, df_test = self.get_train_test_split(train_corpus=self.train_corpus,
                                                      test_corpus=self.test_corpus,
                                                      model_cols=yvar + xvars,
                                                      season=season,
                                                      lookback=self.lookback)
        
        # save outputs here
        df_forecasts = pd.DataFrame(index=df_test.index)
        models = dict()
        
        # iterate over each stat to forecast
        for y in yvar:
                        
            # train model
            models[y] = self.model.fit(X=df_train[xvars],y=df_train[y])
            
            # get forecast
            df_forecasts = df_forecasts.join(self.forecast(df_test=df_test,
                                                           yvar=y,
                                                           xvars=xvars,
                                                           model=models[y],
                                                           pred_type=pred_type))
            
        # merge team + forecast data
        return df_test[["teamID","yearID"]].join(df_forecasts).set_index(["teamID","yearID"])
    

class AggForecasts:
    
    def __init__(self,models,seasons,stats,ensemble,**kwargs):
        
        # init data / params
        self.models = models
        self.seasons = seasons
        self.stats = stats
        self.ensemble = ensemble
        self.kwargs = kwargs
        
    def run_iterator(self,models,seasons,stats):
        
        # append results here
        dft = pd.DataFrame()
        
        # iter over models
        for mdl in tqdm_notebook(models):
            
            # temp df
            temp = pd.DataFrame()
            
            # iter over seasons
            for szn in seasons:
                    
                # get forecasts
                if "mc" not in mdl.model_name:
                    f = mdl.projections(szn,stats,**self.kwargs)
                    
                else:
                    f = mdl.projections(szn,stats)

                # adjust col names
                f.columns = [c + mdl.model_name for c in f.columns]

                # append
                temp = pd.concat([temp,f])
                
            # join temp
            if dft.empty:
                dft = temp.copy()
            else:
                dft = dft.merge(temp,
                                how="outer",
                                left_index=True,
                                right_index=True)
    
        return dft
    
    def get_ensemble(self,df,stats,models):
        
        # copy df
        df = df.copy()
        
        # iterate over each stat
        for st in stats:
            
            # filter for cols
            cols = [st + m.model_name for m in models]
            
            # create ensemble forecast
            df[f"{st}comb"] = df[cols].mean(1)
            
        return df
    
    def agg(self):
        
        # aggregate multiple forecasts
        self.df_agg = self.run_iterator(models=self.models,seasons=self.seasons,stats=self.stats)
        
        # get ensemble forecast
        if self.ensemble:
            self.df_agg = self.get_ensemble(df=self.df_agg,stats=self.stats,models=self.models)
            
        return self
    
    def select_forecast(self,name):
        
        # copy data
        df = self.df_agg[[c for c in self.df_agg.columns if name in c]].copy()
        
        # rename cols
        df.columns = self.stats
        
        return df


class AggByTeam:
    
    def __init__(self,dfb,dfp):
        
        # init data
        self.dfb = dfb
        self.dfp = dfp
        
    def select_player_corpus(self,df,corpus):
        
        # filter by corpus type
        if corpus == "op-day":
            
            # create a temp opening day vector
            opening_day_max = np.where(df["yearID"]==2020, pd.to_datetime(df["yearID"].apply(int).astype(str) + "-08" + "-05"),
                                       pd.to_datetime(df["yearID"].apply(int).astype(str) + "-04" + "-05"))

            # create col indicating player debut on or before opening day
            df["isOpRost"] = (df["debut"] <= opening_day_max) * 1
            
            return df.query("isOpRost == 1")[["playerID","yearID","tmFirst"]].set_index(["playerID","yearID"])
            
        
    def transform_data(self,data_class,corpus,forecast,gross_up,ptype):
        
        # get player data
        pdf = self.select_player_corpus(df=data_class.df_primary,corpus=corpus)
        
        # get forecast data
        fdf = data_class.forecasts.select_forecast(forecast)
        
        # merge data
        pdf = fdf.join(pdf).reset_index()
        
        # rename
        pdf = pdf.rename({"tmFirst":"teamID"},axis=1)
        
        # aggregate by team, year
        pdf =  pdf.assign(playerCount=1).groupby(["teamID","yearID"]).sum()
        
        # gross up to 26 man roster
        if gross_up:
            
            # get roster size
            roster_size = pdf.reset_index()["yearID"].apply(lambda x: 13 if x >= 2021 else 12.5).values
            
            # iterate over cols
            for col in pdf.drop(["playerCount"],axis=1).columns:
                pdf[col] = (roster_size * pdf[col] / pdf["playerCount"])
        
        return pdf.round()
    
    def agg(self,corpus,forecast,gross_up):
        
        # pitcher data
        p = self.transform_data(data_class=self.dfp,corpus=corpus,forecast=forecast,gross_up=gross_up,ptype="P")
        p = p.rename({"R":"RA","H":"HA"},axis=1)
        
        # batting data
        b = self.transform_data(data_class=self.dfb,corpus=corpus,forecast=forecast,gross_up=gross_up,ptype="B")
        
        return b.join(p,lsuffix="B",rsuffix="P")