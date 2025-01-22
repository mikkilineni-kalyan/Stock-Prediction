import numpy as np







import pandas as pd







import talib







from scipy import signal







import logging















logger = logging.getLogger(__name__)















class PatternRecognizer:







    def __init__(self):







        self.patterns = {







            'double_top': self._check_double_top,







            'double_bottom': self._check_double_bottom,







            'head_shoulders': self._check_head_shoulders,







            'triangle': self._check_triangle,







            'wedge': self._check_wedge,







            'channel': self._check_channel







        }







        







    def analyze_patterns(self, data):







        pattern_results = {}







        for pattern_name, pattern_func in self.patterns.items():







            try:







                confidence = pattern_func(data)







                if confidence > 0.7:  # Minimum confidence threshold







                    pattern_results[pattern_name] = {







                        'confidence': confidence,







                        'detected_at': data.index[-1]







                    }







            except Exception as e:







                logger.error(f"Pattern analysis error ({pattern_name}): {str(e)}")







        return pattern_results 







    def _check_double_top(self, data, threshold=0.02):



        try:



            peaks = signal.find_peaks(data['High'].values)[0]



            if len(peaks) >= 2:



                peak1, peak2 = peaks[-2:]



                price1, price2 = data['High'].iloc[peak1], data['High'].iloc[peak2]



                if abs(price1 - price2) / price1 < threshold:



                    return 0.8  # High confidence



            return 0.0



        except Exception as e:



            logger.error(f"Double top check error: {str(e)}")



            return 0.0







    def _check_head_shoulders(self, data):



        try:



            peaks = signal.find_peaks(data['High'].values)[0]



            troughs = signal.find_peaks(-data['Low'].values)[0]



            if len(peaks) >= 3 and len(troughs) >= 2:



                # Check pattern formation



                left_shoulder = data['High'].iloc[peaks[-3]]



                head = data['High'].iloc[peaks[-2]]



                right_shoulder = data['High'].iloc[peaks[-1]]



                if head > left_shoulder and head > right_shoulder:



                    return 0.9  # Very high confidence



            return 0.0



        except Exception as e:



            logger.error(f"Head and shoulders check error: {str(e)}")



            return 0.0







    def _check_triangle(self, data, window=20):



        try:



            highs = data['High'].rolling(window=window).max()



            lows = data['Low'].rolling(window=window).min()



            



            # Check for converging trend lines



            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]



            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]



            



            # Triangle patterns show converging lines



            if abs(high_slope) > 0.001 and abs(low_slope) > 0.001:



                if (high_slope < 0 and low_slope > 0):  # Converging lines



                    return 0.8



            return 0.0



        except Exception as e:



            logger.error(f"Triangle pattern check error: {str(e)}")



            return 0.0







    def _check_wedge(self, data, window=20):



        try:



            closes = data['Close'].values



            trend = np.polyfit(range(len(closes)), closes, 1)[0]



            



            if self._check_triangle(data) > 0:



                # Falling wedge in uptrend



                if trend > 0:



                    return 0.9



                # Rising wedge in downtrend



                elif trend < 0:



                    return 0.9



            return 0.0



        except Exception as e:



            logger.error(f"Wedge pattern check error: {str(e)}")



            return 0.0






