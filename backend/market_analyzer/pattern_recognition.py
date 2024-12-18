import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Pattern:
    name: str
    confidence: float
    direction: str
    description: str
    start_idx: int
    end_idx: int

class AdvancedPatternRecognition:
    def __init__(self):
        self.min_pattern_size = 5
        self.max_pattern_size = 50
        
    def analyze_patterns(self, prices: np.array, volumes: np.array) -> List[Pattern]:
        patterns = []
        
        # Add new pattern detection methods
        patterns.extend(self.detect_head_and_shoulders(prices))
        patterns.extend(self.detect_double_patterns(prices))
        patterns.extend(self.detect_triangles(prices))
        patterns.extend(self.detect_channels(prices))
        patterns.extend(self.detect_flags_pennants(prices))
        patterns.extend(self.detect_cup_and_handle(prices))
        
        # Add volume-based patterns
        patterns.extend(self.detect_volume_breakouts(prices, volumes))
        
        return patterns

    def detect_head_and_shoulders(self, prices: np.array) -> List[Pattern]:
        patterns = []
        window = 20
        
        for i in range(len(prices) - window):
            segment = prices[i:i+window]
            
            # Find local peaks
            peaks = self.find_peaks(segment)
            
            if len(peaks) >= 3:
                # Check for head and shoulders pattern
                left_shoulder = peaks[0]
                head = peaks[1]
                right_shoulder = peaks[2]
                
                if (abs(left_shoulder - right_shoulder) < 0.02 * head and  # Shoulders at similar levels
                    head > left_shoulder and head > right_shoulder):       # Head higher than shoulders
                    
                    patterns.append(Pattern(
                        name="Head and Shoulders",
                        confidence=0.8,
                        direction="bearish",
                        description="Classic reversal pattern indicating potential downward trend",
                        start_idx=i,
                        end_idx=i+window
                    ))
                    
        return patterns

    def detect_cup_and_handle(self, prices: np.array) -> List[Pattern]:
        patterns = []
        window = 30
        
        for i in range(len(prices) - window):
            segment = prices[i:i+window]
            
            # Calculate U-shape for cup
            first_third = segment[:window//3]
            middle_third = segment[window//3:2*window//3]
            last_third = segment[2*window//3:]
            
            if (self.is_decreasing(first_third) and
                self.is_stable(middle_third) and
                self.is_increasing(last_third)):
                
                patterns.append(Pattern(
                    name="Cup and Handle",
                    confidence=0.7,
                    direction="bullish",
                    description="Bullish continuation pattern showing potential upward breakout",
                    start_idx=i,
                    end_idx=i+window
                ))
                
        return patterns

    def detect_flags_pennants(self, prices: np.array) -> List[Pattern]:
        patterns = []
        window = 15
        
        for i in range(len(prices) - window):
            segment = prices[i:i+window]
            
            # Calculate trend lines
            upper_line, lower_line = self.calculate_trend_lines(segment)
            
            # Check for converging/parallel lines
            if self.is_flag_pattern(upper_line, lower_line):
                pattern_type = "Flag" if self.is_parallel(upper_line, lower_line) else "Pennant"
                
                patterns.append(Pattern(
                    name=pattern_type,
                    confidence=0.6,
                    direction="continuation",
                    description=f"{pattern_type} pattern indicating potential trend continuation",
                    start_idx=i,
                    end_idx=i+window
                ))
                
        return patterns

    def detect_volume_breakouts(self, prices: np.array, volumes: np.array) -> List[Pattern]:
        patterns = []
        window = 10
        volume_threshold = 1.5  # 50% above average
        
        for i in range(len(prices) - window):
            price_segment = prices[i:i+window]
            volume_segment = volumes[i:i+window]
            
            avg_volume = np.mean(volume_segment[:-1])
            last_volume = volume_segment[-1]
            
            if last_volume > avg_volume * volume_threshold:
                # Price breakout with volume confirmation
                if price_segment[-1] > np.mean(price_segment[:-1]):
                    patterns.append(Pattern(
                        name="Volume Breakout",
                        confidence=0.75,
                        direction="bullish",
                        description="High volume breakout indicating strong buying pressure",
                        start_idx=i,
                        end_idx=i+window
                    ))
                    
        return patterns

    def calculate_trend_lines(self, prices: np.array) -> Tuple[np.array, np.array]:
        x = np.arange(len(prices))
        
        # Calculate upper trend line
        peaks = self.find_peaks(prices)
        if len(peaks) >= 2:
            upper_coeffs = np.polyfit(peaks, prices[peaks], 1)
            upper_line = np.polyval(upper_coeffs, x)
        else:
            upper_line = np.full_like(prices, np.nan)
            
        # Calculate lower trend line
        troughs = self.find_troughs(prices)
        if len(troughs) >= 2:
            lower_coeffs = np.polyfit(troughs, prices[troughs], 1)
            lower_line = np.polyval(lower_coeffs, x)
        else:
            lower_line = np.full_like(prices, np.nan)
            
        return upper_line, lower_line

    @staticmethod
    def find_peaks(prices: np.array) -> np.array:
        peaks = []
        for i in range(1, len(prices)-1):
            if prices[i-1] < prices[i] > prices[i+1]:
                peaks.append(i)
        return np.array(peaks)

    @staticmethod
    def find_troughs(prices: np.array) -> np.array:
        troughs = []
        for i in range(1, len(prices)-1):
            if prices[i-1] > prices[i] < prices[i+1]:
                troughs.append(i)
        return np.array(troughs)

    @staticmethod
    def is_parallel(line1: np.array, line2: np.array) -> bool:
        slope1 = (line1[-1] - line1[0]) / len(line1)
        slope2 = (line2[-1] - line2[0]) / len(line2)
        return abs(slope1 - slope2) < 0.1

    @staticmethod
    def is_decreasing(prices: np.array) -> bool:
        return np.polyfit(np.arange(len(prices)), prices, 1)[0] < 0

    @staticmethod
    def is_increasing(prices: np.array) -> bool:
        return np.polyfit(np.arange(len(prices)), prices, 1)[0] > 0

    @staticmethod
    def is_stable(prices: np.array) -> bool:
        return abs(np.polyfit(np.arange(len(prices)), prices, 1)[0]) < 0.1 