import React, { useState, useRef } from 'react';
import './UploadContentSection.css';

// PDF processing - using pdf-lib for PDF manipulation
// Note: You'll need to install pdf-lib: npm install pdf-lib
let PDFLib: any = null;
try {
  PDFLib = require('pdf-lib');
} catch (error) {
  console.warn('pdf-lib not available. PDF splitting functionality disabled. Install with: npm install pdf-lib');
}

interface Settings {
  serverUrl: string;
}

interface UploadContentSectionProps {
  settings: Settings;
  onContentUploaded: () => void;
}

const UploadContentSection: React.FC<UploadContentSectionProps> = ({ 
  settings, 
  onContentUploaded 
}) => {
  const [textContent, setTextContent] = useState('');
  const [comments, setComments] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setUploadedFiles(prevFiles => [...prevFiles, ...files]);
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setTextContent('');
    setComments('');
    setUploadedFiles([]);
    setUploadStatus('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        // Remove the data:mime/type;base64, prefix
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = error => reject(error);
    });
  };

  const saveFileToMirixDir = async (file: File, customFilename?: string): Promise<string> => {
    try {
      // Generate unique filename to avoid conflicts
      const timestamp = Date.now();
      const filename = customFilename || `${timestamp}_${file.name}`;
      
      // Convert file to base64 for upload
      const base64 = await fileToBase64(file);
      
      // Save file via backend endpoint - let backend handle the ~/.mirix/files directory
      const response = await fetch(`${settings.serverUrl}/save_uploaded_file`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_name: filename,
          file_data: base64,
          original_name: file.name,
          file_type: file.type
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to save file: ${response.statusText}`);
      }

      const result = await response.json();
      return result.file_path; // Backend returns the full file path
    } catch (error) {
      console.error('Error saving file:', error);
      throw error;
    }
  };

  const splitPdfIntoChunks = async (file: File, pagesPerChunk: number = 5): Promise<File[]> => {
    if (!PDFLib) {
      throw new Error('PDF processing not available. Please install pdf-lib: npm install pdf-lib');
    }

    try {
      const { PDFDocument } = PDFLib;
      
      // Read the PDF file
      const arrayBuffer = await file.arrayBuffer();
      const pdfDoc = await PDFDocument.load(arrayBuffer);
      
      const totalPages = pdfDoc.getPageCount();
      const chunks: File[] = [];
      
      // Split into chunks
      for (let startPage = 0; startPage < totalPages; startPage += pagesPerChunk) {
        const endPage = Math.min(startPage + pagesPerChunk - 1, totalPages - 1);
        
        // Create new PDF document for this chunk
        const newPdf = await PDFDocument.create();
        
        // Copy pages to new document
        for (let pageIndex = startPage; pageIndex <= endPage; pageIndex++) {
          const [copiedPage] = await newPdf.copyPages(pdfDoc, [pageIndex]);
          newPdf.addPage(copiedPage);
        }
        
        // Generate PDF bytes
        const pdfBytes = await newPdf.save();
        
        // Create new File object for this chunk
        const chunkNumber = Math.floor(startPage / pagesPerChunk) + 1;
        const totalChunks = Math.ceil(totalPages / pagesPerChunk);
        const originalName = file.name.replace('.pdf', '');
        const chunkName = `${originalName}_part${chunkNumber}of${totalChunks}_pages${startPage + 1}-${endPage + 1}.pdf`;
        
        const chunkFile = new File([pdfBytes], chunkName, { type: 'application/pdf' });
        chunks.push(chunkFile);
      }
      
      return chunks;
    } catch (error) {
      console.error('Error splitting PDF:', error);
      throw new Error(`Failed to split PDF: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const checkVideoSupport = (file: File): boolean => {
    const video = document.createElement('video');
    const canPlay = video.canPlayType(file.type);
    console.log(`[DEBUG] Video support check for ${file.type}: "${canPlay}"`);
    
    // Log more detailed support information
    if (canPlay === 'probably') {
      console.log(`[DEBUG] Browser strongly supports ${file.type}`);
    } else if (canPlay === 'maybe') {
      console.log(`[DEBUG] Browser might support ${file.type} - will attempt processing`);
    } else {
      console.log(`[DEBUG] Browser does not support ${file.type} - will fall back to file upload`);
    }
    
    // Clean up the test video element
    video.src = '';
    video.load();
    return canPlay !== '';
  };

  const processVideoFrameByFrame = async (file: File, videoName: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      console.log(`[DEBUG] Starting video processing for: ${file.name}`);
      console.log(`[DEBUG] File type: ${file.type}, size: ${file.size}`);
      
      // Check if browser supports this video format
      if (!checkVideoSupport(file)) {
        reject(new Error(`Unsupported video format: ${file.type}. Please try converting to a standard MP4 format.`));
        return;
      }
      
      const video = document.createElement('video');
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        console.error('[DEBUG] Failed to get canvas context');
        reject(new Error('Failed to get canvas context'));
        return;
      }

      let currentTime = 0;
      let frameNumber = 0;
      let isProcessing = false;
      let objectURL: string | null = null;
      let hasErrored = false; // Prevent multiple error callbacks
      let timeoutId: NodeJS.Timeout | null = null;
      
      // Cleanup function
      const cleanup = () => {
        console.log(`[DEBUG] Cleaning up video resources for ${file.name}`);
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        if (objectURL) {
          URL.revokeObjectURL(objectURL);
          objectURL = null;
        }
        video.src = '';
        video.load(); // Reset the video element
        // Remove all event listeners to prevent memory leaks
        video.onloadedmetadata = null;
        video.onseeked = null;
        video.onerror = null;
        video.onloadstart = null;
        video.onloadeddata = null;
        video.oncanplay = null;
      };


      
      video.onloadstart = () => {
        console.log(`[DEBUG] Video loading started: ${file.name}`);
        setUploadStatus(`Loading video: ${file.name}...`);
      };
      
      video.onloadeddata = () => {
        console.log('[DEBUG] Video data loaded');
      };
      
      video.oncanplay = () => {
        console.log('[DEBUG] Video can start playing');
      };
      
      // Single error handler for the video element (handles both loading and processing errors)
      video.onerror = (event) => {
        if (hasErrored) {
          console.log('[DEBUG] Error handler already called, ignoring subsequent errors');
          return;
        }
        hasErrored = true;
        
        console.error('[DEBUG] Video error:', event);
        console.log(`[DEBUG] Video readyState: ${video.readyState}, networkState: ${video.networkState}`);
        
        // Get more detailed error information
        const error = (video as any).error;
        let errorMessage = `Failed to process video: ${file.name}`;
        
        if (error) {
          console.log(`[DEBUG] Video error details - code: ${error.code}, message: ${error.message}`);
          switch (error.code) {
            case 1: // MEDIA_ERR_ABORTED
              errorMessage += ' (playback aborted)';
              break;
            case 2: // MEDIA_ERR_NETWORK
              errorMessage += ' (network error)';
              break;
            case 3: // MEDIA_ERR_DECODE
              errorMessage += ' (decoding error - unsupported codec or corrupted file)';
              break;
            case 4: // MEDIA_ERR_SRC_NOT_SUPPORTED
              errorMessage += ' (unsupported format)';
              break;
            default:
              errorMessage += ` (error code: ${error.code})`;
          }
          if (error.message) {
            errorMessage += ` - ${error.message}`;
          }
        } else {
          console.log('[DEBUG] No detailed error information available');
          errorMessage += ' (no error details available)';
        }
        
        cleanup();
        reject(new Error(errorMessage));
      };
      
      // Set video properties for better compatibility
      // Note: Don't set crossOrigin for local files as it can cause security issues
      video.preload = 'metadata';
      video.muted = true; // Muted videos can autoplay in browsers
      video.playsInline = true; // For mobile compatibility
      
      // Add a timeout to prevent hanging
      timeoutId = setTimeout(() => {
        if (!hasErrored) {
          hasErrored = true;
          console.error('[DEBUG] Video loading timeout after 30 seconds');
          cleanup();
          reject(new Error(`Video loading timeout: ${file.name} took too long to load`));
        }
      }, 30000); // 30 second timeout
      
      // Clear timeout on successful load
      video.onloadedmetadata = async () => {
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        console.log(`[DEBUG] Video metadata loaded - duration: ${video.duration}s, dimensions: ${video.videoWidth}x${video.videoHeight}`);
        
        if (video.duration <= 0 || isNaN(video.duration)) {
          console.error('[DEBUG] Invalid video duration');
          cleanup();
          reject(new Error('Invalid video duration'));
          return;
        }
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const duration = video.duration;
        const totalFrames = Math.floor(duration); // 1 frame per second

        
        const processNextFrame = async () => {
          if (currentTime >= duration) {
            console.log(`[DEBUG] Video processing completed - processed ${frameNumber} frames`);

            if (timeoutId) {
              clearTimeout(timeoutId);
              timeoutId = null;
            }
            cleanup();
            resolve();
            return;
          }
          
          if (isProcessing) {
            console.warn('[DEBUG] Already processing a frame, skipping...');
            return;
          }
          
          console.log(`[DEBUG] Seeking to time: ${currentTime}s`);
          video.currentTime = currentTime;
        };
        
        video.onseeked = async () => {
          if (isProcessing) return;
          isProcessing = true;
          
          try {
            frameNumber++;
            console.log(`[DEBUG] Processing frame ${frameNumber} at time ${currentTime}s`);
            
            // Extract current frame
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const frameDataURL = canvas.toDataURL('image/jpeg', 0.8);
            const base64Data = frameDataURL.split(',')[1];
            
            if (!base64Data || base64Data.length === 0) {
              throw new Error('Failed to extract frame data');
            }
            
            console.log(`[DEBUG] Frame ${frameNumber} extracted, base64 length: ${base64Data.length}`);
            
            setUploadStatus(`Processing frame ${frameNumber}/${Math.floor(duration)} from ${file.name}...`);

            
            // Save frame to backend
            console.log(`[DEBUG] Saving frame ${frameNumber} to backend...`);
            const imageUri = await saveFrameToMirixTmp(base64Data, frameNumber, videoName);
            console.log(`[DEBUG] Frame ${frameNumber} saved, imageUri: ${imageUri}`);
            
            // Send frame to agent with force_absorb_content: false
            const frameMessage = `Video frame ${frameNumber} from ${file.name} (timestamp: ${frameNumber}s)`;
            console.log(`[DEBUG] Sending frame ${frameNumber} to agent...`);
            const success = await sendSingleRequest(frameMessage, [], [], imageUri, false);
            
            if (!success) {
              throw new Error(`Failed to process frame ${frameNumber} from ${file.name}`);
            }
            
            console.log(`[DEBUG] Frame ${frameNumber} successfully processed`);
            
            // Move to next second and process next frame
            currentTime += 1;
            isProcessing = false;
            
            // Small delay before processing next frame
            setTimeout(() => {
              processNextFrame();
            }, 500);
            
          } catch (error) {
            console.error(`[DEBUG] Error processing frame ${frameNumber}:`, error);
            isProcessing = false;
            cleanup();
            reject(error);
          }
        };
        
        // Note: The main error handler is already set up before loading
        // It will handle errors during both loading and processing phases
        
        // Start processing
        processNextFrame();
      };
      
      // Start loading the video
      try {
        objectURL = URL.createObjectURL(file);
        console.log(`[DEBUG] Created object URL: ${objectURL}`);
        console.log(`[DEBUG] File size: ${file.size} bytes, MIME type: ${file.type}`);
        video.src = objectURL;
        
        // Force load the video
        video.load();
      } catch (error) {
        cleanup();
        console.error('[DEBUG] Error creating object URL:', error);
        reject(new Error(`Failed to create video URL: ${error instanceof Error ? error.message : 'Unknown error'}`));
      }
    });
  };

  const saveFrameToMirixTmp = async (frameBase64: string, frameNumber: number, videoName: string): Promise<string> => {
    try {
      const timestamp = Date.now();
      const filename = `${videoName}_frame_${frameNumber}_${timestamp}.jpg`;
      
      console.log(`[DEBUG] Saving frame to backend: ${filename}`);
      console.log(`[DEBUG] Server URL: ${settings.serverUrl}`);
      console.log(`[DEBUG] Frame data length: ${frameBase64.length}`);
      
      // Save frame to ~/.mirix/tmp/images via backend endpoint
      const response = await fetch(`${settings.serverUrl}/save_video_frame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          frame_data: frameBase64,
          filename: filename,
          video_name: videoName,
          frame_number: frameNumber
        })
      });

      console.log(`[DEBUG] Backend response status: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[DEBUG] Backend error response: ${errorText}`);
        throw new Error(`Failed to save frame: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      console.log(`[DEBUG] Backend response:`, result);
      
      if (!result.image_uri) {
        throw new Error('Backend did not return image_uri');
      }
      
      return result.image_uri; // Backend returns the image URI
    } catch (error) {
      console.error('[DEBUG] Error saving frame:', error);
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error(`Network error: Cannot connect to backend at ${settings.serverUrl}. Is the server running?`);
      }
      throw error;
    }
  };

  const sendSingleRequest = async (
    message: string, 
    voiceFiles: string[] = [], 
    filePaths: string[] = [],
    imageUri?: string,
    forceAbsorbContent: boolean = true
  ): Promise<boolean> => {
    console.log(`[DEBUG] Sending request to agent:`);
    console.log(`[DEBUG] - Message: ${message}`);
    console.log(`[DEBUG] - Image URI: ${imageUri}`);
    console.log(`[DEBUG] - Force absorb content: ${forceAbsorbContent}`);
    
    const requestBody: any = {
      message: message,
      memorizing: true,
      force_absorb_content: forceAbsorbContent
    };

    if (voiceFiles.length > 0) {
      requestBody.voice_files = voiceFiles;
    }
    if (filePaths.length > 0) {
      requestBody.file_paths = filePaths;
    }
    if (imageUri) {
      requestBody.image_uri = imageUri;
    }

    console.log(`[DEBUG] Request body:`, requestBody);

    const response = await fetch(`${settings.serverUrl}/send_streaming_message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    console.log(`[DEBUG] Agent response status: ${response.status} ${response.statusText}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[DEBUG] Agent error response: ${errorText}`);
      throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
    }

    // Handle streaming response
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('Failed to get response reader');
    }

    let finalResult = false;
    let success = false;
    
    while (!finalResult) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const eventData = JSON.parse(line.slice(6));
            
            switch (eventData.type) {
              case 'intermediate':
                setUploadStatus(`${eventData.message_type}: ${eventData.content}`);
                break;
                
              case 'final':
                success = true;
                finalResult = true;
                break;
                
              case 'missing_api_keys':
                setUploadStatus(`Missing API keys for ${eventData.model_type}: ${eventData.missing_keys?.join(', ') || 'Unknown keys'}`);
                finalResult = true;
                break;
                
              case 'error':
                setUploadStatus(`Error: ${eventData.error}`);
                finalResult = true;
                break;
            }
          } catch (parseError) {
            console.warn('Failed to parse SSE data:', line, parseError);
          }
        }
      }
    }

    return success;
  };

  const submitContent = async () => {
    if (!textContent.trim() && uploadedFiles.length === 0) {
      setUploadStatus('Please provide some content or upload files');
      return;
    }

    setIsUploading(true);
    setUploadStatus('Preparing content...');

    try {
      // Separate files by type
      const pdfFiles = uploadedFiles.filter(file => 
        file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
      );
      const videoFiles = uploadedFiles.filter(file => {
        const isVideoByType = file.type.startsWith('video/');
        const isVideoByExtension = /\.(mp4|mov|avi|webm|ogv|mkv)$/i.test(file.name);
        return isVideoByType || isVideoByExtension;
      });
      const otherFiles = uploadedFiles.filter(file => {
        const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
        const isVideoByType = file.type.startsWith('video/');
        const isVideoByExtension = /\.(mp4|mov|avi|webm|ogv|mkv)$/i.test(file.name);
        return !isPdf && !isVideoByType && !isVideoByExtension;
      });

      // Process non-PDF, non-MP4 files first
      const voiceFiles: string[] = [];
      const filePaths: string[] = [];

      for (const file of otherFiles) {
        setUploadStatus(`Saving ${file.name}...`);
        
        if (file.type.startsWith('audio/') || file.name.toLowerCase().includes('voice')) {
          // For voice files, send as base64
          const base64 = await fileToBase64(file);
          voiceFiles.push(base64);
        } else {
          // For all other files (including images), save to ~/.mirix/files and get file path
          const filePath = await saveFileToMirixDir(file);
          filePaths.push(filePath);
        }
      }

      // Send initial content (text + non-PDF, non-MP4 files) if any
      if (textContent.trim() || comments.trim() || otherFiles.length > 0) {
        let message = '';
        if (textContent.trim()) {
          message += `Content: ${textContent.trim()}`;
        }
        if (comments.trim()) {
          message += `\n\nComments: ${comments.trim()}`;
        }
        if (otherFiles.length > 0) {
          message += `\n\nUploaded ${otherFiles.length} non-PDF, non-video file(s) to be processed from local storage.`;
        }

        setUploadStatus('Sending initial content to agent...');
        await sendSingleRequest(message, voiceFiles, filePaths);
      }

      // Process video files - upload as complete files (no frame extraction)
      for (let videoIndex = 0; videoIndex < videoFiles.length; videoIndex++) {
        const videoFile = videoFiles[videoIndex];
        setUploadStatus(`Uploading video ${videoIndex + 1}/${videoFiles.length}: ${videoFile.name}...`);
        
        try {
          console.log(`[DEBUG] Uploading video file: ${videoFile.name}`);
          const videoFilePath = await saveFileToMirixDir(videoFile);
          const message = `Video file: ${videoFile.name} (uploaded as complete video file for AI analysis)`;
          const success = await sendSingleRequest(message, [], [videoFilePath]);
          
          if (success) {
            setUploadStatus(`✅ Successfully uploaded ${videoFile.name} as video file`);
          } else {
            throw new Error(`Failed to upload video file ${videoFile.name}`);
          }
          
        } catch (videoError) {
          console.error(`Error uploading video ${videoFile.name}:`, videoError);
          setUploadStatus(`❌ Error: Could not upload ${videoFile.name} - ${videoError instanceof Error ? videoError.message : 'Upload failed'}`);
          // Continue with other files instead of stopping completely
        }
      }

      // Process PDF files - split each into chunks and send separately
      for (let pdfIndex = 0; pdfIndex < pdfFiles.length; pdfIndex++) {
        const pdfFile = pdfFiles[pdfIndex];
        setUploadStatus(`Processing PDF ${pdfIndex + 1}/${pdfFiles.length}: ${pdfFile.name}...`);
        
        try {
          // Split PDF into 5-page chunks
          const pdfChunks = await splitPdfIntoChunks(pdfFile, 5);
          
          setUploadStatus(`Split ${pdfFile.name} into ${pdfChunks.length} chunks. Processing...`);
          
          // Send each chunk separately
          for (let chunkIndex = 0; chunkIndex < pdfChunks.length; chunkIndex++) {
            const chunk = pdfChunks[chunkIndex];
            setUploadStatus(`Processing ${chunk.name} (${chunkIndex + 1}/${pdfChunks.length})...`);
            
            // Save chunk to backend
            const chunkPath = await saveFileToMirixDir(chunk, chunk.name);
            
            // Send chunk to agent
            const chunkMessage = `PDF chunk from ${pdfFile.name}: ${chunk.name}`;
            const success = await sendSingleRequest(chunkMessage, [], [chunkPath]);
            
            if (!success) {
              throw new Error(`Failed to process chunk ${chunk.name}`);
            }
            
            // Small delay between chunks to avoid overwhelming the agent
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
          
        } catch (pdfError) {
          console.error(`Error processing PDF ${pdfFile.name}:`, pdfError);
          
          // Fallback: send the whole PDF as a single file
          setUploadStatus(`Failed to split ${pdfFile.name}, sending as whole file...`);
          const wholePdfPath = await saveFileToMirixDir(pdfFile);
          const wholePdfMessage = `Complete PDF file: ${pdfFile.name} (could not split - processing as whole document)`;
          await sendSingleRequest(wholePdfMessage, [], [wholePdfPath]);
        }
      }

      // All done
      setUploadStatus('All content successfully added to memory!');
      clearAll();
      onContentUploaded(); // Trigger memory visualization refresh
      
      // Clear status after 3 seconds
      setTimeout(() => {
        setUploadStatus('');
      }, 3000);

    } catch (error) {
      console.error('Error uploading content:', error);
      setUploadStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);

    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="upload-content-section">
      <h2>Upload Content</h2>
      
      {/* Text Content Input */}
      <div className="input-group">
        <label>Text Content</label>
        <textarea
          value={textContent}
          onChange={(e) => setTextContent(e.target.value)}
          placeholder="Enter text content (news, notes, information, etc.)"
          rows={6}
          className="text-content-input"
        />
      </div>

      {/* File Upload */}
      <div className="input-group">
        <label>Upload Files</label>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={handleFileSelect}
          className="file-input"
          accept=".txt,.pdf,.doc,.docx,.jpg,.jpeg,.png,.gif,.mp4,.mov,.avi,.webm,.ogv,.mkv"
        />
        <div className="file-upload-hint">
          <small>
            📄 <strong>PDF files</strong> will be automatically split into 5-page chunks for better processing.<br/>
            🎬 <strong>Video files</strong> (MP4/MOV/AVI/WebM/etc.) will be uploaded as complete video files for AI analysis.<br/>
            🎵 Audio files are sent as voice data. 📷 Images and other documents are saved as files.
          </small>
        </div>
        {uploadedFiles.length > 0 && (
          <div className="uploaded-files">
            <h4>Selected Files:</h4>
            {uploadedFiles.map((file, index) => (
              <div key={index} className="file-item">
                <span className="file-name">{file.name}</span>
                <span className="file-size">({formatFileSize(file.size)})</span>
                <button 
                  onClick={() => removeFile(index)}
                  className="remove-file-btn"
                  type="button"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </div>



      {/* Comments */}
      <div className="input-group">
        <label>Comments</label>
        <textarea
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          placeholder="Add your comments or notes about this content"
          rows={3}
          className="comments-input"
        />
      </div>

      {/* Action Buttons */}
      <div className="action-buttons">
        <button
          onClick={submitContent}
          disabled={isUploading || (!textContent.trim() && uploadedFiles.length === 0)}
          className="submit-btn"
        >
          {isUploading ? 'Processing...' : 'Add to Memory'}
        </button>
        <button
          onClick={clearAll}
          disabled={isUploading}
          className="clear-btn"
        >
          Clear All
        </button>
      </div>

      {/* Status Message */}
      {uploadStatus && (
        <div className={`status-message ${uploadStatus.includes('Error') ? 'error' : 'success'}`}>
          {uploadStatus}
        </div>
      )}
    </div>
  );
};

export default UploadContentSection; 