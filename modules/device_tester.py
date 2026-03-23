"""
Audio Device Tester
Finds audio input devices that are capturing active audio.

Based on streaming/device_tester.py
"""

import sys
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

from utils.logger import setup_logger

logger = setup_logger("device_tester")


# Device names to SKIP (microphones, hardware mics)
SKIP_PATTERNS = [
    'mic', 'microphone', 'input', 'filter', 'equalizer',
    'alc257', 'hw:', 'sysdefault', 'pipewire'
]

# Device names to PRIORITIZE (playback monitors, sinks)
PRIORITY_PATTERNS = [
    'monitor', 'sink', 'source', 'brave', 'firefox', 
    'chrome', 'output', 'easy effects'
]

PEAK_THRESHOLD = 0.01  # Minimum peak to consider as "has audio"


def should_skip_device(name: str) -> bool:
    """Check if device should be skipped based on name."""
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in SKIP_PATTERNS)


def is_priority_device(name: str) -> bool:
    """Check if device is a priority (playback) device."""
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in PRIORITY_PATTERNS)


def list_input_devices(include_all: bool = False) -> List[Dict[str, Any]]:
    """
    Get list of input devices.
    
    Args:
        include_all: If True, include mics/filters too
        
    Returns:
        List of input devices with index, name, channels, sample_rate
    """
    if not SOUNDDEVICE_AVAILABLE:
        return []
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            name = dev['name']
            
            if not include_all and should_skip_device(name):
                continue
            
            input_devices.append({
                'index': i,
                'name': name,
                'channels': dev['max_input_channels'],
                'sample_rate': int(dev['default_samplerate']),
                'priority': is_priority_device(name)
            })
    
    # Sort: priority devices first
    input_devices.sort(key=lambda x: (not x['priority'], x['index']))
    
    return input_devices


def test_device(device_index: int, duration: float = 1.0) -> Dict[str, Any]:
    """
    Test a single audio device for audio activity.
    
    Args:
        device_index: Device index to test
        duration: Duration to record in seconds
        
    Returns:
        dict with: device, success, peak, has_audio, error
    """
    if not SOUNDDEVICE_AVAILABLE:
        return {'device': device_index, 'success': False, 'error': 'sounddevice not available'}
    
    result = {
        'device': device_index,
        'success': False,
        'peak': 0.0,
        'has_audio': False,
        'error': None
    }
    
    try:
        device_info = sd.query_devices(device_index)
        sample_rate = int(device_info['default_samplerate'])
        channels = min(device_info['max_input_channels'], 2)
        
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            device=device_index,
            dtype=np.float32
        )
        sd.wait()
        
        if recording is not None and len(recording) > 0:
            if recording.ndim > 1:
                audio = recording.mean(axis=1)
            else:
                audio = recording.flatten()
            
            peak = float(np.max(np.abs(audio)))
            result['success'] = True
            result['peak'] = peak
            result['has_audio'] = peak >= PEAK_THRESHOLD
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def find_working_device(test_duration: float = 1.0) -> Optional[int]:
    """
    Test all input devices and find one with audio.
    
    Args:
        test_duration: Duration to test each device
        
    Returns:
        Device index with audio, or None if none found
    """
    logger.info("Finding working audio device...")
    
    devices = list_input_devices()
    
    if not devices:
        logger.error("No input devices found")
        return None
    
    logger.info(f"Found {len(devices)} input devices")
    
    working_devices = []
    
    for dev in devices:
        idx = dev['index']
        name = dev['name']
        
        logger.info(f"Testing [{idx}] {name}...")
        
        result = test_device(idx, test_duration)
        
        if result['error']:
            logger.warning(f"  Error: {result['error']}")
        elif result['has_audio']:
            logger.info(f"  ✅ Audio detected! Peak: {result['peak']:.4f}")
            working_devices.append({
                'index': idx,
                'name': name,
                'peak': result['peak']
            })
        else:
            logger.info(f"  Silent (peak: {result['peak']:.6f})")
    
    if working_devices:
        # Sort by peak (highest first)
        working_devices.sort(key=lambda x: x['peak'], reverse=True)
        best = working_devices[0]
        logger.info(f"Recommended device: {best['index']} ({best['name']})")
        return best['index']
    else:
        logger.warning("No devices detected audio")
        return None


def get_default_device() -> Optional[int]:
    """Get default input device index."""
    if not SOUNDDEVICE_AVAILABLE:
        return None
    
    try:
        default = sd.default.device[0]  # Input device
        return default if default >= 0 else None
    except:
        return None
