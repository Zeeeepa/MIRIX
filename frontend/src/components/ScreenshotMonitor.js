import React, { useState, useRef, useCallback } from 'react';
// import VoiceRecorder from './VoiceRecorder';
import './ScreenshotMonitor.css';
import queuedFetch from '../utils/requestQueue';

const ScreenshotMonitor = ({ settings, onMonitoringStatusChange }) => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [screenshotCount, setScreenshotCount] = useState(0);
  const [lastProcessedTime, setLastProcessedTime] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [skipSimilarityCheck, setSkipSimilarityCheck] = useState(false);
  const [isRequestInProgress, setIsRequestInProgress] = useState(false);
  const [isProcessingScreenshot, setIsProcessingScreenshot] = useState(false);
  const [hasScreenPermission, setHasScreenPermission] = useState(null);
  const [isCheckingPermission, setIsCheckingPermission] = useState(false);
  
  // Voice recording state - COMMENTED OUT
  // const [voiceData, setVoiceData] = useState([]);
  // const voiceRecorderRef = useRef(null);
  
  const intervalRef = useRef(null);
  const lastImageDataRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Configuration (matches main.py defaults)
  const INTERVAL = 2000; // 2 seconds (changed from 1 second)
  const SIMILARITY_THRESHOLD = 0.99;

  // Check screenshot permissions
  const checkScreenPermissions = useCallback(async () => {
    if (!window.electronAPI || !window.electronAPI.takeScreenshot) {
      setHasScreenPermission(false);
      setError('Screenshot functionality is only available in the desktop app');
      return false;
    }

    setIsCheckingPermission(true);
    setError(null);

    try {
      // Try to take a test screenshot to check permissions
      const result = await window.electronAPI.takeScreenshot();
      
      if (result.success) {
        setHasScreenPermission(true);
        // Clean up the test screenshot
        if (result.filepath) {
          try {
            await window.electronAPI.deleteScreenshot(result.filepath);
          } catch (cleanupError) {
            // Silent cleanup error
          }
        }
        return true;
      } else {
        setHasScreenPermission(false);
        if (result.error && result.error.includes('permission')) {
          setError('Screen recording permission not granted. Please grant screen recording permissions in System Preferences > Security & Privacy > Screen Recording and restart the application.');
        } else {
          setError(result.error || 'Failed to access screenshot functionality');
        }
        return false;
      }
    } catch (err) {
      setHasScreenPermission(false);
      if (err.message && err.message.includes('permission')) {
        setError('Screen recording permission not granted. Please grant screen recording permissions in System Preferences > Security & Privacy > Screen Recording and restart the application.');
      } else {
        setError(`Permission check failed: ${err.message}`);
      }
      return false;
    } finally {
      setIsCheckingPermission(false);
    }
  }, []);



  // Open System Preferences to Screen Recording section
  const openSystemPreferences = useCallback(async () => {
    if (!window.electronAPI || !window.electronAPI.openScreenRecordingPrefs) {
      setError('System Preferences functionality is only available in the desktop app');
      return;
    }

    try {
      const result = await window.electronAPI.openScreenRecordingPrefs();
      if (result.success) {
        setError(null);
        // Check permissions again after a short delay to see if they were granted
        setTimeout(() => {
          checkScreenPermissions();
        }, 2000);
      } else {
        setError(result.message || 'Failed to open System Preferences');
      }
    } catch (err) {
      setError(`Failed to open System Preferences: ${err.message}`);
    }
  }, [checkScreenPermissions]);

  // Handle voice data from the recorder - COMMENTED OUT
  // const handleVoiceData = useCallback((data) => {
  //   setVoiceData(prev => [...prev, data]);
  // }, []);

  // Calculate image similarity using a simple pixel difference approach
  // Note: This is a simplified version compared to SSIM in main.py
  const calculateImageSimilarity = useCallback((imageData1, imageData2) => {
    if (!imageData1 || !imageData2) return 0;
    if (imageData1.length !== imageData2.length) return 0;

    let totalDiff = 0;
    const pixelCount = imageData1.length / 4; // RGBA channels

    for (let i = 0; i < imageData1.length; i += 4) {
      // Calculate grayscale values for comparison
      const gray1 = 0.299 * imageData1[i] + 0.587 * imageData1[i + 1] + 0.114 * imageData1[i + 2];
      const gray2 = 0.299 * imageData2[i] + 0.587 * imageData2[i + 1] + 0.114 * imageData2[i + 2];
      
      totalDiff += Math.abs(gray1 - gray2);
    }

    const averageDiff = totalDiff / pixelCount;
    const similarity = 1 - (averageDiff / 255); // Normalize to 0-1 range
    return Math.max(0, Math.min(1, similarity));
  }, []);

  // Convert canvas to image data for similarity comparison
  const getImageDataFromCanvas = useCallback((canvas) => {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return imageData.data;
  }, []);

  // Delete screenshot that is too similar - only call this after backend processing is complete
  const deleteSimilarScreenshot = useCallback(async (filepath) => {
    if (!window.electronAPI || !window.electronAPI.deleteScreenshot) {
      return;
    }

    try {
      await window.electronAPI.deleteScreenshot(filepath);
    } catch (error) {
      // Silent error handling
    }
  }, []);

  // Send screenshot to backend with memorizing=true and accumulated audio - VOICE FUNCTIONALITY COMMENTED OUT
  const sendScreenshotToBackend = useCallback(async (screenshotFile) => {
    if (!screenshotFile || isRequestInProgress) {
      return;
    }

    let currentAbortController = null;
    let cleanup = null;

    try {
      setIsRequestInProgress(true);
      setStatus('sending');
      
      // Get accumulated audio from voice recorder - COMMENTED OUT
      // let accumulatedAudio = [];
      // let voiceFiles = [];
      
      // if (voiceRecorderRef.current && typeof voiceRecorderRef.current.getAccumulatedAudio === 'function') {
      //   accumulatedAudio = voiceRecorderRef.current.getAccumulatedAudio();
      // }

      // Convert audio blobs to base64 for sending with screenshot - COMMENTED OUT
      // if (accumulatedAudio.length > 0) {
      //   try {
      //     for (const audioData of accumulatedAudio) {
      //       const arrayBuffer = await audioData.blob.arrayBuffer();
      //       const base64Data = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      //       voiceFiles.push(base64Data);
      //     }
      //   } catch (audioError) {
      //     // Silent error handling
      //   }
      // }

      // Prepare the message with voice context info - COMMENTED OUT
      // let message = null;
      // if (voiceFiles.length > 0) {
      //   const totalDuration = accumulatedAudio.reduce((sum, audio) => sum + audio.duration, 0);
      //   message = `[Screenshot with voice recording: ${voiceFiles.length} audio chunks, ${(totalDuration/1000).toFixed(1)}s total]`;
      // }

      const requestData = {
        // message: message,
        image_uris: [screenshotFile.path],
        // voice_files: voiceFiles.length > 0 ? voiceFiles : null, // COMMENTED OUT
        memorizing: true, // This is the key difference from chat
        is_screen_monitoring: true // Indicate this request is from screen monitoring
      };

      // Use a fresh abort controller for this request
      currentAbortController = new AbortController();
      abortControllerRef.current = currentAbortController;

      const result = await queuedFetch(`${settings.serverUrl}/send_streaming_message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal: currentAbortController.signal,
        isStreaming: true
      });

      const response = result.response;
      cleanup = result.cleanup;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Increment count immediately after successful response
      setScreenshotCount(prev => prev + 1);
      setLastProcessedTime(new Date().toISOString());
      setStatus('monitoring');
      setError(null);

      // Consume the streaming response to complete the request
      // This is done after incrementing the count to ensure count updates even if streaming fails
      try {
        if (response.body) {
          const reader = response.body.getReader();
          while (true) {
            const { done } = await reader.read();
            if (done) break;
          }
        }
      } catch (streamError) {
        // Log streaming error but don't fail the whole request since we already counted it
        console.warn('Error consuming streaming response:', streamError);
      }

      // Clear accumulated audio after successful send - COMMENTED OUT
      // if (voiceRecorderRef.current && typeof voiceRecorderRef.current.clearAccumulatedAudio === 'function') {
      //   voiceRecorderRef.current.clearAccumulatedAudio();
      // }

      return { success: true, shouldDelete: false };

    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Failed to send screenshot:', err);
        setError(`Failed to send screenshot: ${err.message}`);
      }
      return { success: false, shouldDelete: true };
    } finally {
      setIsRequestInProgress(false);
      // Clear the abort controller if it's still the current one
      if (abortControllerRef.current?.signal === currentAbortController?.signal) {
        abortControllerRef.current = null;
      }
      
      // Call cleanup to notify request queue
      if (cleanup) {
        cleanup();
      }
    }
  }, [settings.serverUrl, isRequestInProgress]);

  // Take and process a screenshot
  const processScreenshot = useCallback(async () => {
    if (!window.electronAPI || !window.electronAPI.takeScreenshot) {
      setError('Screenshot functionality requires desktop app');
      return;
    }

    // Skip if already processing a screenshot or if a request is in progress
    if (isProcessingScreenshot || isRequestInProgress) {
      return;
    }

    try {
      setIsProcessingScreenshot(true);
      setStatus('capturing');

      // Take screenshot
      const result = await window.electronAPI.takeScreenshot();
      
      if (!result.success) {
        throw new Error(result.error || 'Failed to take screenshot');
      }

      const screenshotFile = {
        name: result.filename,
        path: result.filepath,
        type: 'image/png',
        size: result.size,
        isScreenshot: true
      };

      // If similarity check is disabled, send every screenshot
      if (skipSimilarityCheck) {
        const sendResult = await sendScreenshotToBackend(screenshotFile);
        if (sendResult && sendResult.shouldDelete) {
          await deleteSimilarScreenshot(screenshotFile.path);
        }
        setStatus('monitoring');
        return;
      }

      // Read image as base64 for similarity comparison
      const imageResult = await window.electronAPI.readImageAsBase64(result.filepath);
      
      if (!imageResult.success) {
        // If we can't read the image for comparison, just send it
        const sendResult = await sendScreenshotToBackend(screenshotFile);
        if (sendResult && sendResult.shouldDelete) {
          await deleteSimilarScreenshot(screenshotFile.path);
        }
        setStatus('monitoring');
        return;
      }

      // Create a temporary canvas to get image data for similarity comparison
      const img = new Image();
      img.onload = async () => {
        // Check again here since this is async and may execute much later
        if (isRequestInProgress) {
          setStatus('monitoring');
          setIsProcessingScreenshot(false);
          return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        const currentImageData = getImageDataFromCanvas(canvas);
        
        // Check similarity with last image
        let similarity = 0;
        if (lastImageDataRef.current) {
          similarity = calculateImageSimilarity(lastImageDataRef.current, currentImageData);
        }

        // Only send if different enough (below threshold)
        if (similarity < SIMILARITY_THRESHOLD) {
          const sendResult = await sendScreenshotToBackend(screenshotFile);
          if (sendResult && sendResult.success) {
            // Update last image data only if sending was successful
            lastImageDataRef.current = currentImageData;
          }
          if (sendResult && sendResult.shouldDelete) {
            await deleteSimilarScreenshot(screenshotFile.path);
          }
        } else {
          // Delete the screenshot since it's too similar
          await deleteSimilarScreenshot(screenshotFile.path);
          setStatus('monitoring');
        }
        setIsProcessingScreenshot(false);
      };

      img.onerror = async () => {
        // If image loading fails, just send the screenshot anyway
        const sendResult = await sendScreenshotToBackend(screenshotFile);
        if (sendResult && sendResult.shouldDelete) {
          await deleteSimilarScreenshot(screenshotFile.path);
        }
        setStatus('monitoring');
        setIsProcessingScreenshot(false);
      };

      // Use the base64 data URL instead of file:// URL
      img.src = imageResult.dataUrl;

    } catch (err) {
      setError(`Error processing screenshot: ${err.message}`);
      setStatus('monitoring');
      setIsProcessingScreenshot(false);
    }
  }, [calculateImageSimilarity, getImageDataFromCanvas, sendScreenshotToBackend, deleteSimilarScreenshot, skipSimilarityCheck, isRequestInProgress, isProcessingScreenshot]);

  // Start monitoring
  const startMonitoring = useCallback(async () => {
    if (isMonitoring) return;

    // Check permissions first
    const hasPermission = await checkScreenPermissions();
    if (!hasPermission) {
      return;
    }

    setIsMonitoring(true);
    setStatus('monitoring');
    setError(null);
    setScreenshotCount(0);
    lastImageDataRef.current = null;
    
    // Notify parent component about monitoring status change
    if (onMonitoringStatusChange) {
      onMonitoringStatusChange(true);
    }

    // Start the interval
    intervalRef.current = setInterval(processScreenshot, INTERVAL);

    // Take first screenshot immediately
    processScreenshot();
  }, [isMonitoring, processScreenshot, checkScreenPermissions, onMonitoringStatusChange]);

  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    if (!isMonitoring) return;

    setIsMonitoring(false);
    setStatus('idle');

    // Clear interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Abort any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Clear image data reference
    lastImageDataRef.current = null;
    
    // Reset request and processing state
    setIsRequestInProgress(false);
    setIsProcessingScreenshot(false);
    
    // Notify parent component about monitoring status change
    if (onMonitoringStatusChange) {
      onMonitoringStatusChange(false);
    }
  }, [isMonitoring, onMonitoringStatusChange]);

  // Check permissions on mount
  React.useEffect(() => {
    checkScreenPermissions();
  }, [checkScreenPermissions]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      stopMonitoring();
    };
  }, [stopMonitoring]);

  const getStatusIcon = () => {
    switch (status) {
      case 'monitoring': return '👁️';
      case 'capturing': return '📸';
      case 'sending': return '📤';
      default: return '⏹️';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'monitoring': return '#28a745';
      case 'capturing': return '#ffc107';
      case 'sending': return '#17a2b8';
      default: return '#6c757d';
    }
  };

  return (
    <div className="screenshot-monitor">
      <div className="monitor-header">
        <h3>🎯 Screen Monitor</h3>
        <div className="monitor-controls">
          <label className="similarity-toggle">
            <input
              type="checkbox"
              checked={skipSimilarityCheck}
              onChange={(e) => setSkipSimilarityCheck(e.target.checked)}
              disabled={isMonitoring}
            />
            <span>Send all screenshots (skip similarity check)</span>
          </label>
          {hasScreenPermission === false && (
            <button
              className="open-prefs-button"
              onClick={openSystemPreferences}
              disabled={false}
              style={{
                backgroundColor: '#dc3545',
                color: 'white',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '4px',
                                  cursor: 'pointer',
                marginRight: '8px'
              }}
            >
              ⚙️ Open System Preferences
            </button>
          )}
          <button
            className={`monitor-toggle ${isMonitoring ? 'active' : ''}`}
            onClick={isMonitoring ? stopMonitoring : startMonitoring}
            disabled={hasScreenPermission === false}
            style={{
              backgroundColor: isMonitoring ? '#dc3545' : hasScreenPermission === false ? '#6c757d' : '#28a745',
              color: 'white',
                              cursor: hasScreenPermission === false ? 'not-allowed' : 'pointer'
            }}
          >
            {hasScreenPermission === false ? '🔒 Permission Required' :
             isMonitoring ? '⏹️ Stop Monitor' : '▶️ Start Monitor'}
          </button>
        </div>
      </div>

      <div className="monitor-status">
        <div className="status-item">
          <span className="status-icon" style={{ color: getStatusColor() }}>
            {getStatusIcon()}
          </span>
          <span className="status-text">
            Status: <strong style={{ color: getStatusColor() }}>{status}</strong>
          </span>
        </div>
        
        <div className="status-item">
          <span className="permission-status">
            📋 Permissions: <strong style={{ 
              color: hasScreenPermission === true ? '#28a745' : 
                     hasScreenPermission === false ? '#dc3545' : '#ffc107' 
            }}>
              {isCheckingPermission ? '⏳ Checking...' :
               hasScreenPermission === true ? '✅ Granted' : 
               hasScreenPermission === false ? '❌ Denied' : '⏳ Checking...'}
            </strong>
          </span>
        </div>
        
        <div className="status-item">
          <span>📊 Screenshots sent: <strong>{screenshotCount}</strong></span>
        </div>
        
        {lastProcessedTime && (
          <div className="status-item">
            <span>🕒 Last sent: <strong>{new Date(lastProcessedTime).toLocaleTimeString()}</strong></span>
          </div>
        )}
      </div>

      {error && (
        <div className="monitor-error">
          ⚠️ {error}
          {error.includes('permission') && (
            <div className="permission-help" style={{ marginTop: '8px', fontSize: '14px', color: '#6c757d' }}>
              <strong>How to grant permission:</strong> 
              <br />1. Click "⚙️ Open System Preferences" button above
              <br />2. Find "MIRIX" in the list and check the box next to it
              <br />3. No restart required - permissions take effect immediately
            </div>
          )}
        </div>
      )}

      {hasScreenPermission === false && !error && (
        <div className="monitor-warning" style={{ 
          backgroundColor: '#fff3cd', 
          color: '#856404', 
          padding: '12px', 
          borderRadius: '4px', 
          border: '1px solid #ffeaa7',
          marginTop: '12px'
        }}>
          🔒 Screen recording permission is required to use the screen monitor feature. 
          <br />
          <strong>Click "⚙️ Open System Preferences" to grant permission directly!</strong>
        </div>
      )}

      {/* Voice Recording Component - COMMENTED OUT */}
      {/* <VoiceRecorder 
        ref={voiceRecorderRef}
        settings={settings}
        isMonitoring={isMonitoring}
        onVoiceData={handleVoiceData}
      /> */}
    </div>
  );
};

export default ScreenshotMonitor; 