# -*- coding: utf-8 -*-
"""
Class to extract features from a timeserie
"""

__author__ = ["SemPhares"]

__all__ = ["get_time_series_features",
            "transform_time_series",
            "distance_based",
            "extract_features_from_many_series",
            "drop_unique_values_columns",
            "extract_distance_from_many_series",
            "return_null_columns",
            "drop_many_columns"
            ]


import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import acf, pacf
from kats.tsfeatures.tsfeatures import TsFeatures
from typing import Tuple, Union, List, Dict

import warnings
warnings.simplefilter(action='ignore')


"--------------------------- Many Other functions ---------------------------"
def timer(function_to_time):
  """
  a decorator to time the functions
  """
  import time
  from functools  import wraps

  @wraps(function_to_time)
  def wraper(*args, **kwargs):
    start = time.time()
    result = function_to_time(*args, **kwargs)
    end = time.time()
    print("\n Runtime : ", round(end-start,3),'secs')

    return result
  
  return wraper


def drop_many_columns(data:pd.DataFrame,
                       col_to_drop:List[str]
                      ) -> pd.DataFrame:

  """
  Drop a list of columns from a table

  Args:
      data : 
      col_to_drop : columns list o drop

  Returns:
      new data frame
  """
  temp = data.copy()
  return temp.drop(col_to_drop, axis=1)


def return_null_columns(
                        data:pd.DataFrame
                        ) -> pd.DataFrame:

  """
  Return the list of columns that contains any (can be modify to all) null or na values from a table

  Args:
      data : 
      
  Returns:
      list of null columns
  """  
  return data.columns[data.isnull().any()].tolist()


def drop_unique_values_columns( data:pd.DataFrame,
                                n_unique: int = 2
                                ) -> pd.DataFrame:
  """
  Drop columns that have only a given number of unique values 

  Args:
      data : 
      n_unique : number of unique values to consider

  Returns:
      new data frame
  """
  temp = data.copy()
  to_drop = []
  for col in temp.columns:
    if len(temp[col].unique()) <= n_unique:
      to_drop.append(col)

  return temp.drop(to_drop,axis=1)

@timer
def extract_features_from_many_series(
                            series: Union[List[Union[list,np.array]],
                                        Dict[str,Union[list,np.array]]],
                            drop_unique_values:bool = False
                            ) -> pd.DataFrame:

    """
    Extract features for many series one shot

    Args:
        series : list of series to extract data from
        drop_unique_values : if drop features with only one unique value

    Returns:
        features dataframe
    """
    all_features = pd.DataFrame( dtype= float)

    if isinstance(series,dict):

        for serie in tqdm(list(series.values())):
            current_feats = get_time_series_features(serie
                                                        ).get_all_features()
            all_features = all_features.append(current_feats)
        
        all_features['names'] = list(series.keys())

    else:
        for serie in tqdm(series):
            current_feats = get_time_series_features(serie
                                                        ).get_all_features()
            all_features = all_features.append(current_feats)

    all_features.index = list(range(0,len(series)))
    
    return drop_unique_values_columns(all_features) \
                            if drop_unique_values else all_features


def method_exists(obj, 
                  method:str
                ) -> bool:
  """
  Check if a given method exists in a class

  Args:
      obj : the class to check in
      method : the method to check for

  Returns:
      bool
  """
  return callable(getattr(obj, method, None))


@timer
def extract_distance_from_many_series(
                            series: List[Union[list,np.array]],
                            distance_methods: List[str]
                            ) -> pd.DataFrame:

    """
    Extract distances for many series one shot

    Args:
        series : list of series to extract data from
        distance_methods : distance m??thods list

    Returns:
        distance dataframe
    """

    all_distances = pd.DataFrame(dtype= float)
    # all_distances = np.zeros((len(series),len(series)))
    if distance_methods == '_all':
      pass
      # too long to compute, need to optimize it
      # cross_ = []
      # for i in tqdm(range(len(series))):
      #     for j in range(i, len(series)):
      #         if i == j :
      #             continue
      #         else: 
      #           current_distance = distance_based(series[i],
      #                                             series[j]).get_all_distances()
                
      #           all_distances = all_distances.append(current_distance)
      #           cross_.append(str(i)+'_'+str(j))

      # all_distances.index = list(range(0,len(series)))
      # all_distances["cross"] = cross_
    
    else:
      for distance in distance_methods:
        if method_exists(distance_based, distance):
          temp_dist = pd.DataFrame( index=range(len(series)),
                                    columns=range(len(series)), 
                                    dtype= float
                                    )

          for i in tqdm(range(len(series))):
            for j in range(i, len(series)):
                if i == j :
                    continue
                else: 
                  # print(i,j)
                  distance_object = distance_based(series[i], series[j],
                                                    alias_1= f'serie_{i}',
                                                    alias_2=f'serie_{j}'
                                                    )
                  distance_function = getattr(distance_object,
                                              distance)

                  temp_dist[i][j] =  distance_function()

          temp_dist.columns = ['serie_'+str(i) for i in range(len(series))]
          temp_dist.insert(0,'row', ['serie_'+str(i) for i in range(len(series))])
          temp_dist.index = [distance]*len(series)
          
          all_distances = all_distances.append(temp_dist)

        else:
          print(distance, "method not found in distance_based class")
          continue

    return all_distances

"--------------------------- End Others functions ---------------------------"



class load_data():

    def __init__(self,
                data_path: str = "_Csv_Data/" ,
                viz_path: str = "_Vis_Data/",
                anno_file_path: str = "Annotation/annotation_full_parVideo.xlsx"
                ) -> None:
        """
        Args:
            data_path: str : path to the dat folder
            viz_path: str : path to the data visualization
            anno_file_path:str : full path to the annotation file
        """
        self.data_path = data_path
        self.viz_path = viz_path
        self.annotations = pd.read_excel(anno_file_path)


    def get_serie_from_path(self,
                            path:str,
                            ) -> Union[np.ndarray,str]:
        """
        Function to extract serie as numpy.array from ArticulationRate data

        Args:
            path:str : path to the ArticulationRate data

        Return:
            serie
        """
        try:
            serie = pd.read_csv(path, names =["time",'ArticulationRate'],skiprows=1)
            serie = serie.ArticulationRate.values
        except FileNotFoundError as f:
            print(f)
            serie = ''
        return serie


    def check_path( self,
                    path:str,
                    ) -> str:
        """
        Check for the existence of the file

        Args:
            path:str : path to the existence for

        Return:
            chekced path
        """
        return path if os.path.exists(path) else ''

        
    def extact_all_series(self) -> pd.DataFrame:
        """
        Function to extract serie for all the csv files

        Return:
            pd.DataFrame object containing the data
        """
        data = self.annotations[['sub_video', 'Gender', 'Persuasiveness_rms',
                        'PerceivedSelf-Confidence_rms', 'AudienceEngagement_rms',
                        'GlobalEvaluation_rms'
                        ]].copy()
        series = []
        images = []
        for name in data.sub_video:
            path = self.data_path+name+"_articulationRate_PolyModel_.csv"
            series.append(self.get_serie_from_path(path))

            image_path = self.viz_path+name+"ArticulationRate__lengS_15.0_nbS_165__sw_1_poly_deg_30.png"
            images.append(self.check_path(image_path))
        
        data['serie'] = series
        data['viz'] = images

        return data
    

    def __call__(self) -> pd.DataFrame:
        
        return self.extact_all_series()


class get_time_series_features():

    def __init__(self,
                 ts: Union[np.array, list]
                 ) -> None:
        """
        Args:
            ts: Union[np.array, list] : the timeserie
        """

        self.ts = self._convert_to_np_array(ts)
        self.times = list(range(1,len(ts)+1))
        self.length = len(ts)


    def _convert_to_np_array(self, 
                            ts :Union[np.array, list]
                            ) -> np.ndarray:
        """
        Function to convert a serie to numpy array format

        Args:
            ts: Union[np.array, list] :   the serie to convert

        Returns:
            Converted time serie
        """
        return np.array(ts) if not isinstance(ts,np.ndarray) else ts


    def plot_ts(self):

        figure = plt.figure(figsize=(20,10))
        plt.plot(self.times, self.ts)
        plt.title("initial time serie")
        plt.show()


    def mean(self) -> float:
        """
        Compute the mean
        
        Returns:
            The mean value of the serie
        """
        return np.mean(self.ts)
    

    def std(self) -> float:
        """
        Compute the standard dev

        Returns:
            The standart-deviation of the serie
        """
        return np.std(self.ts)


    def var(self) -> float:
        """
        Compute the variance

        Returns:
            The variance of the serie
        """
        return np.var(self.ts)


    def cor(self) -> float:
        """
        Compute the correlation coef

        Returns:
            The correlation value of the serie
        """
        return np.corrcoef(self.ts)


    def max(self) -> float:
        """
        Compute the max

        Returns:
            The maximum value of the serie
        """
        return np.max(self.ts)


    def min(self) -> float:
        """
        Compute the min

        Returns:
            The minimum value of the serie
        """
        return np.min(self.ts)


    def mad(self) -> float:
        """
        Compute median absolute deviation

        Note:
            The Median Absolute Deviation (MAD) is a measure of dispersion or variability in a dataset. 
            It is a robust statistic, meaning that it is less sensitive to outliers or extreme values in the data than other measures of dispersion like the standard deviation.
            The MAD is often used as a robust alternative to the standard deviation, especially in cases where the data may contain outliers or is not normally distributed.
            One limitation of the MAD is that it is less efficient than the standard deviation for normally distributed data.

        Returns:
            The median absolute deviation value of the serie
        """
        med = np.median(self.ts)
        
        return np.median(np.abs(self.ts - med))


    def sma(self) -> float:
        """ 
        Compute the Signal magnitude area

        Note:
            Signal magnitude area is a measure of the total magnitude of a signal over a specified period of time. 
            It is used to provide a summary measure of the signal magnitude, especially in the case of non-stationary signals 
            where the mean and standard deviation are not appropriate measures of central tendency and variability.

        Returns:
            The signal magnitude area value of the serie
        """
        return np.sqrt(np.mean(self.ts**2))


    def energy(self) -> float:
        """ 
        Compute Energy measure : Sum of the squares divided by the number of values.

        Returns:
            The measure of energy of the serie
        """
        return (np.mean(self.ts**2))


    def iqr(self) -> float:
        """ 
        Compute the Interquartile range of the serie

        Returns:
            The diffferecne between 75th percentile and the 25th percentile the serie
        """
        q1, q3 = np.percentile(self.ts, [25, 75])
        return q3 - q1


    def max_magnitude_index(self) -> int :
        """ 
        Calculate the index of the frequency component with largest magnitude

        Returns:
            The index of the maximum magnitude from Fast Fourier Transform
        """
        fff = np.abs(np.fft.fft(self.ts))
        return np.argmax(fff)


    def skewness(self) -> float:
        """ 
        Compute the skewness value of the serie based on scipy.stats module

        Returns:
            The skewness of the frequency domain signal
        """
        return skew(self.ts, axis=0, bias=True)

    def return_kurtosis(self) -> float:
        """ 
        Compute the kurtosis of the the serie based on scipy.stats module

        Returns:
            The kurtosis value of the frequency domain signal
        """
        return kurtosis(self.ts, axis=0, bias=True)


    def entropy(self) -> float:
        """ 
        Compute the entropy of the serie

        Note:
            The entropy of a time series measure the amount of randomness in the data.
            It is a scalar value that provides a measure of how predictable or unpredictable the data is.
            A time series with a high entropy value indicates that the values in the series are spread out and have high variability, 
            whereas a low entropy value means that the values are clustered together and have low variability.

        Returns:
            The entropy value

        """
        _, counts = np.unique(self.ts, return_counts=True)
        prob = counts / len(self.ts)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy


    def _slope(self, 
                ts:Union[np.array, list]
                ) -> float:
        """
        Compute the slope of a time series using linear regression

        Args:
            ts: Union[np.array, list] :  The serie to convert

        Returns:
            The slope value of the serie
        """
        times = list(range(1,len(ts)+1))
        mean_times = np.mean(times)
        mean_serie = np.mean(ts)
        
        numerator = sum([(t - mean_times) * (v - mean_serie) for t, v in zip(times,ts)])
        denominator = sum([(t - mean_times)**2 for t in times])

        if denominator!=0:
            result = numerator / denominator
        else :
            result = 0
        
        return result


    def slope(self) -> float:
        """
        Return the slope from the _slope function
        """
        return self._slope(self.ts)


    def amplitude(self) -> float:
        """
        Compute the amplutude based on the Fast Fourier Transform

        Returns:
            The index of the maximum value of the Fast Fourier Transform serie
        """
        fourier = np.abs(np.fft.fft(self.ts))
        return max(fourier)

        
    def get_acf_features(self,
                         lag: int = 6
                         ) -> pd.DataFrame :
        """
        Calculate ACF based features via statsmodels
        Args:
            lag : Number of lags to return ACF/PACF features for
        Returns:
            ACF features.
        """
        assert len(np.unique(self.ts)) !=1 ,'The time serie is constant, ACF or PACF features can not be calculated'

        columns = ['acf_'+str(i) for i in range(1,lag+1)]
        acf_features = acf(self.ts,fft=True, nlags=lag)[1:]

        return  pd.DataFrame(acf_features.reshape(1,lag),columns=columns, dtype= float)


    def get_pacf_features(self,
                          lag: int = 6,
                          ) -> pd.DataFrame:
        """
        Calculate PACF based features via statsmodels
        Args:
            lag : Number of lags to return ACF/PACF features for
        Returns:
            PACF features.
        """
        assert len(np.unique(self.ts)) !=1 ,'The time serie is constant, ACF or PACF features can not be calculated'

        columns = ['pacf_'+str(i) for i in range(1,lag+1)]
        pacf_features = pacf(self.ts, nlags=lag)[1:]
        
        return pd.DataFrame(pacf_features.reshape(1,lag),columns=columns, dtype= float)



    def extract_kats_features(self, 
                              start_date: str = "2023-01-01"
                              ) -> pd.DataFrame:
        """
        Extract features from the series based on the facebook kats api

        Note:
            TsFeatures is a module for performing adhoc feature engineering on time series data using different statistic
            https://facebookresearch.github.io/Kats/
            https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_203_tsfeatures.ipynb

            on linux machine preferably
            pip install -q packaging==21.3
            pip install -q kats

        Args:
            start_date:str : usefull to transform the data in correct shape for the TsFeatures algorithm
        
        Returns
            Dataframe of 40 features
        """
        
        t_series = pd.DataFrame(self.ts, columns=['value'])
        t_series.insert(0,'time', pd.date_range(start_date, periods=self.length, freq="s"))
        t_series.columns = ['time', 'value']

        #load the features generation model without time features
        selected_features = [
              'acfpacf_features', 'holt_params', 'hw_params', 'level_shift_features',
               'nowcasting','seasonalities', 'special_ac', 'statistics', 'stl_features'
                        ]
              # , 'trend_detector', 'cusum_detector','outlier_detector',
              #   'bocp_detector',  'robust_stat_detector', 
        model = TsFeatures(selected_features = selected_features)
        #extract the features from the timeserie
        output_features = model.transform(t_series)
        output_features = pd.DataFrame(output_features,index=[0])

        return output_features


    def get_all_features(self) -> pd.DataFrame:
        """
        Method to return all the feature at once

        Retunrs:
            Aggregrate features
        """

        methods = ['amplitude',
                    'energy',
                    'entropy',
                    'iqr',
                    'mad',
                    'max_magnitude_index',
                    'min',
                    'kurtosis',
                    'skewness',
                    'slope',
                    'sma',
                    'std'
                    ]

        values = np.array(
                [   self.amplitude(),
                    self.energy(),
                    self.entropy(),
                    self.iqr(),
                    self.mad(),
                    self.max_magnitude_index(),
                    self.min(),
                    self.return_kurtosis(),
                    self.skewness(),
                    self.slope(),
                    self.sma(),
                    self.std() 
                ])

        all_features = pd.DataFrame(values.reshape(1,len(methods)),columns=methods, dtype= float)

        acf_feats = self.get_acf_features()
        pacf_feats = self.get_pacf_features()
        kats_feats = self.extract_kats_features()
        kats_feats.rename(columns = {'entropy':'entropie_de_Shannon_normalis??e'},inplace =True)


        all_features = pd.concat([all_features, 
                                  acf_feats.iloc[:,1:], 
                                  pacf_feats.iloc[:,1:],
                                  kats_feats.iloc[:,1:]
                                ], axis=1)

        return all_features




class transform_time_series():

    def __init__(self,
                 ts: Union[np.array, list]
                 ) -> None:
        """
        Args:
            ts: Union[np.array, list] : the timeserie
        """
        
        self.ts = self._convert_to_np_array(ts)
        self.times = list(range(1,len(ts)+1))
        self.length = len(ts)


    def _convert_to_np_array(self, 
                            ts :Union[np.array, list]
                            ) -> np.ndarray:
        """
        Function to convert a serie to numpy array format

        Args:
            ts: Union[np.array, list] :   the serie to convert

        Returns:
            Converted time serie
        """
        return np.array(ts) if not isinstance(ts,np.ndarray) else ts


    def fourier_transform(self) -> np.ndarray:
        """
        Compute the fast Fourier Transform

        Note:
            The Fourier transform allows to represent in frequency signals which are not periodic

        Returns:
            The Fast Fourier transformed serie
        """    
        return np.fft.fft(self.ts)


    def plot_ts_and_fourier_transform(self):
        """
        Plot fourier transform and the time serie

        Returns:
            The plot to compare the original serie and the transformed serie
        """
        fourier = self.fourier_transform()

        figure = plt.figure(figsize=(20,10))

        plt.subplot(2, 1, 1)
        plt.plot(self.ts)
        plt.title("initial time serie")

        plt.subplot(2, 1, 2)
        plt.plot(fourier,c='red')
        plt.title("fourier transform")

        plt.show()         

        
    # def extract_shapelets(self, 
    #                         num_shapelets: int = 6 , 
    #                         shapelet_length: int = 20
    #                         ) -> List[List[float]]:
    #     """
    #     Compute shaplet extraction from time series

    #     Args:
    #        num_shapelets : number of shapelet to extract
    #        shapelet_length : max length of shapelet

    #     Returns:
    #         List of shapelets        
    #     """
    #     ts = self.ts.copy()
    #     shapelets = []
    #     for i in range(num_shapelets):
    #         best_shapelet = None
    #         best_dist = float('inf')
    #         for j in range(len(ts) - shapelet_length + 1):
    #             shapelet = ts[j:j + shapelet_length]
    #             dist = np.sum((ts - shapelet) ** 2)
    #             if dist < best_dist:
    #                 best_dist = dist
    #                 best_shapelet = shapelet
    #         shapelets.append(best_shapelet)
    #         ts = np.delete(ts, np.arange(best_dist, best_dist + shapelet_length))

    #     return shapelets


    def _choose_intervals(self, 
                          ts: Union[np.array, list], 
                          nbr_intervals: int = 3 ,
                          min_intervals_len: int = 10
                          ) -> List[Tuple[int,int]]:
        """" 
        Return random intervals of a timeserie timeline

        Args:
            ts : The time serie to split
            nbr_intervals : the number of interval wanted
            min_intervals_len : the minimal length of each interval

        Returns:
            List of timeline intervals 
        """
        length = len(ts)
        times = list(range(1,len(ts)+1))

        if nbr_intervals*min_intervals_len >= length:
            raise ValueError(
                'nbr_intervals*len_intervals = {} is >= or = {}. Please consider another values'.format(
                                                                nbr_intervals*min_intervals_len,length)
                            )
        else:
            intervals = []
            while len(intervals) < nbr_intervals:
                start = random.randint(0, length - min_intervals_len)
                end = start + min_intervals_len + random.randint(0, length - start - min_intervals_len)
                # make sure there are not overlaping
                intervals.append((times[start], times[end]))

            return intervals


    def _split_equals_length_intervals(self,
                                       ts: Union[np.array, list],
                                       nbr_intervals: int = 3
                                       ) -> List[Union[list, np.ndarray]]:
        """ 
        Split a time serie in equals length intervals

        Args:
            ts : The time serie to split
            nbr_intervals : the number of interval wanted

        Returns:
            List of intervals 
        """
        if nbr_intervals >= len(ts) :
            raise ValueError ('PLease consider lower nbr_intervals value')
        else:
            timeline_len = len(ts)
            subseq_len = timeline_len // nbr_intervals

        return [ts[i:i+subseq_len] for i in range(0, timeline_len, subseq_len)]


    def _return_3_features(self,
                           ts: Union[np.array, list]
                           ) -> List[float]:
        """ 
        Return the mean, std and slope for a timeserie

        Args:
            ts : The time serie

        Returns:
            List of mean, std and slope
        """
        features = []
        features.append(np.mean(ts)) 
        features.append(np.std(ts)) 
        features.append(get_time_series_features(ts).slope())
        return features


    def _time_forest_features(self, 
                              ts: Union[np.array, list], 
                              nbr_intervals: int = 3, 
                              min_intervals_len: int = 10
                              ) -> list:
        """
        Extract features based on time forest methods

        Args:
            ts : The time serie
            nbr_intervals : the number of interval wanted
            min_intervals_len : the minimal length of each interval

        Returns:
            List of features
        """
        intervals = self._choose_intervals(ts, nbr_intervals, min_intervals_len)

        features = [] 
        for index, interval in enumerate(intervals):
            temp = ts[interval[0]:interval[1]+1]
            features.append(self._return_3_features(temp))

        return [feats for elem in features for feats in elem]


    def time_forest_features(self, 
                             nbr_intervals: int = 3, 
                             min_intervals_len: int = 10
                             ) -> list:
        """
        Extract features based on time forest methods for the self time serie

        Args:
            nbr_intervals : the number of interval wanted
            min_intervals_len : the minimal length of each interval

        Returns:
            List of features 
        """
        return self._time_forest_features(self.ts, nbr_intervals, min_intervals_len)


    def bag_of_features(self,
                        nbr_intervals: int = 3, 
                        min_intervals_len: int = 20
                        ) -> list:
        """
        Extract features based on bag of features methods

        Args:
            nbr_intervals : the number of interval wanted
            min_intervals_len : the minimal length of each interval

        Returns:
            List of features 
        """

        intervals = self._choose_intervals(self.ts, nbr_intervals, min_intervals_len)
        features = []

        for interval in intervals:
            temp = self.ts[interval[0]:interval[1]+1]
            
            # append features of the current interval
            features.append(
                    self._return_3_features(temp)
                )
            # split the current interval into equals length subsequence
            subsequence_temp = self._split_equals_length_intervals(temp)

            # append features of each subsequence of the current interval
            for sequence in subsequence_temp:
                features.append(
                    self._return_3_features(sequence)
                )

        return [feats for elem in features for feats in elem]     




class distance_based():

    def __init__(self, 
                 ts1: Union[np.array, list], 
                 ts2: Union[np.array, list],
                 alias_1: str = 'serie_1',
                 alias_2: str = 'serie_2',
                 ) -> None:
        """
        Args:
            ts1: Union[np.array, list] : the fisrt timeserie
            ts2: Union[np.array, list] : the second tiemserie
            alias_1: str = 'serie_1' : alias for the plot of ts1 
            alias_2: str = 'serie_2' : alias for the plot of ts2
        """
        self.ts1 = self.normalize(np.array(ts1, dtype=float))
        self.times1 = list(range(1,len(ts1)+1))
        self.length1 = len(ts1)
        self.alias_1 = alias_1

        self.ts2 = self.normalize(np.array(ts2, dtype=float))
        self.times2 = list(range(1,len(ts2)+1))
        self.length2 = len(ts2)
        self.alias_2 = alias_2

    def normalize(self,
                  ts: Union[np.array, list]
                  ) -> np.ndarray :

        """
        Transform ts to have [0,1] range

        Args:
            ts: Union[np.array, list] :  the serie to normalize

        Returns:
            Normalized time serie
        """
        return (ts - ts.min()) / (ts.max() - ts.min())

    def plot_them(self):
        """
        Plot the two timeseries

        Returns:
            The plot to compare the two series
        """
        figure = plt.figure(figsize=(20,10))

        plt.subplot(2, 1, 1)
        plt.plot(self.ts1)
        plt.title(self.alias_1)

        plt.subplot(2, 1, 2)
        plt.plot(self.ts2,c='red')
        plt.title(self.alias_2)

        plt.show()  


    def longuest_common_subsequence(self,
                                    epsilon:float = 0.1
                                    ) -> float :

        """
        Compute the LCSS distance between ts1 and ts2

        Args:
            epsilon : the two series composants matching threshold

        Returns:
            lcss : The longuest common subsequence of the two series
            score : The longuest common subsequence score bewteen the two series 
        """
        #initiate the common sequence matrix
        s = np.zeros((self.length1+1, self.length2+1))

        # looping trhough the series
        for i in range(self.length1 + 1):
            for j in range(self.length2 + 1):
                if i == 0 or j == 0:
                    s[i][j] = 0
                elif np.abs(self.ts1[i-1]-self.ts2[j-1]) <= epsilon :
                    s[i][j] = s[i - 1][j - 1] + 1
                else:
                    s[i][j] = max(s[i - 1][j], s[i][j - 1])

        lcss = int(s[self.length1][self.length2])
        score = lcss/max(self.length1 ,self.length2)   
        return score
        

    def dtw(self) -> float:
        """
        Compute the dtw distance between ts1 and ts2

        Returns:
            DTW distance value
        """
        dtw = np.full((self.length1+1, self.length2+1), np.inf)
        dtw[0][0] = 0
        for i in range(1, self.length1 + 1):
            for j in range(1, self.length2 + 1):
                cost = np.sum(np.abs(self.ts1[i - 1] - self.ts2[j - 1]))
                dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])

        max_len = max(self.length1, self.length2)

        #to have a similarity value between 0 and 1 we use
        # with 1 indicating a perfect match and 0 indicating a completely dissimilar match
        return 1 - dtw[self.length1][self.length2] / max_len  
                # return dtw[self.length1][self.length2]
        

    def edit_distance(self) -> float:
        """
        Compute the edit distance between ts1 and ts2

        Returns:
            The edit distance value
        """
        d = np.zeros((self.length1+1, self.length2+1))
        for i in range(self.length1 + 1):
            for j in range(self.length2 + 1):
                if i == 0:
                    d[i][j] = j
                elif j == 0:
                    d[i][j] = i
                elif self.ts1[i - 1] == self.ts2[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i][j - 1], d[i - 1][j], d[i - 1][j - 1])
        
        max_len = max(self.length1, self.length2)
        return 1 - d[self.length1][self.length2]/ max_len
                # return d[self.length1][self.length2]


    def _padify(self) -> Tuple[np.ndarray]:
        """
        Add pad 0 to the shortest series so the both could have the same length

        Returns:
            Zero-Padifyed timeseries
        """

        s1, s2 = self.ts1.copy() , self.ts2.copy()

        if self.length1 < self.length2:
            s1 = np.pad(self.ts1, (0, self.length2 - self.length1), mode='constant')
        elif self.length2 < self.length1:
            s2 = np.pad(self.ts2, (0, self.length1 - self.length2), mode='constant')

        return s1,s2


    def euclidean_distance(self) -> float:
        """
        Compute Euclidean distance

        Returns:
            Euclidean distance between the two series
        """
        s1, s2 = self._padify()
        return np.sqrt(np.sum((s1 - s2)**2))


    def correlation_distance(self) -> float:
        """
        Compute Corelation distance

        Returns:
            The Corelation distance of the two series
        """
        s1, s2 = self._padify()
            
        mean1 = np.mean(self.ts1)
        mean2 = np.mean(self.ts2)
        std1 = np.std(self.ts1)
        std2 = np.std(self.ts2)
        corr = np.sum((s1 - mean1) * (s2 - mean2)) / (std1 * std2 * self.length1)
        return 1 - corr


    def cumulative_differences_distance(self) -> float:
        """
        Compute measurement based on the average of the cumulative differences

        Returns:
            The cumulative difference mean as distance
        """
        s1, s2 = self._padify()

        cumulative_diff = np.abs(np.cumsum(s1) - np.cumsum(s2))
        mean_ = np.mean(cumulative_diff)
        return mean_


    def get_all_distances(self) -> pd.DataFrame:
        """
        Method to return all the computed distance at once

        Retunrs:
            Aggregrate distance between the two serie
        """

        distances = [
                'correlation_distance',
                'cumulative_differences_distance',
                'dtw',
                'edit_distance',
                'euclidean_distance',
                'longuest_common_subsequence'
                ]

        values = np.array(
                [   self.correlation_distance(),
                    self.cumulative_differences_distance(),
                    self.dtw(),
                    self.edit_distance(),
                    self.euclidean_distance(),
                    self.longuest_common_subsequence()
                ])

        all_distance = pd.DataFrame(values.reshape(1,len(distances)), columns=distances, dtype = float, index=[0])

        return all_distance

