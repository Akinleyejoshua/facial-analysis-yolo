import { useRef, useState, useEffect } from 'react';
import { Camera, Square, Loader2, Play, Pause } from 'lucide-react';
import * as ort from 'onnxruntime-web';

interface Detection {
  classId: number;
  score: number;
  bbox: [number, number, number, number];
  className: string;
}

const COCO_CLASSES = ["angry", "fear", "disgust", "happy", "sad", "neutral", "surprise"]

const CLASS_COLORS = COCO_CLASSES.map((_, i) => {
  const hue = (i * 137.5) % 360;
  return `hsl(${hue}, 70%, 50%)`;
});

const IMG_SIZE = 128

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [error, setError] = useState<string>('');
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });

  const sessionRef = useRef<any>(null);
  const animationRef = useRef<number>(0);
  const fpsIntervalRef = useRef<number>(0);
  const frameCountRef = useRef(0);

  useEffect(() => {
    loadModel();
    startCamera();

    return () => {
      stopDetection();
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const updateVideoSize = () => {
      if (videoRef.current) {
        setVideoSize({
          width: videoRef.current.clientWidth,
          height: videoRef.current.clientHeight
        });
      }
    };

    const video = videoRef.current;
    if (video) {
      video.addEventListener('loadedmetadata', updateVideoSize);
      window.addEventListener('resize', updateVideoSize);
    }

    return () => {
      if (video) {
        video.removeEventListener('loadedmetadata', updateVideoSize);
      }
      window.removeEventListener('resize', updateVideoSize);
    };
  }, []);

  const loadModel = async () => {
    try {
      setIsLoading(true);
      setError('');

      // @ts-ignore
      if (!ort) {
        throw new Error('ONNX Runtime not loaded');
      }

      sessionRef.current = await ort.InferenceSession.create('best.onnx', {
        executionProviders: ['wasm']
      });

      setModelLoaded(true);
      setIsLoading(false);
    } catch (err) {
      setError('Failed to load model. Please ensure best.onnx is in the public folder.');
      setIsLoading(false);
      console.error(err);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 640 }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      setError('Failed to access camera');
      console.error(err);
    }
  };

  const preprocessImage = (video: HTMLVideoElement): Float32Array => {
    const canvas = document.createElement('canvas');
    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;
    const ctx = canvas.getContext('2d')!;

    ctx.drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE);
    const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    const pixels = imageData.data;

    const input = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
    for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
      input[i] = pixels[i * 4] / 255.0;
      input[IMG_SIZE * IMG_SIZE + i] = pixels[i * 4 + 1] / 255.0;
      input[2 * IMG_SIZE * IMG_SIZE + i] = pixels[i * 4 + 2] / 255.0;
    }

    return input;
  };

  const processDetections = (output: any): Detection[] => {
    const detections: Detection[] = [];
    const dims = output.dims;
    const data = output.data;

    console.log('Output dims:', dims);
    console.log('Output data length:', data.length);

    const [_, numOutputs, numDetections] = dims;
    const numClasses = numOutputs - 4;

    console.log('Num classes:', numClasses, 'Num detections:', numDetections);

    for (let i = 0; i < numDetections; i++) {
      const xCenter = data[i];
      const yCenter = data[numDetections + i];
      const width = data[2 * numDetections + i];
      const height = data[3 * numDetections + i];

      let maxScore = 0;
      let maxClassId = 0;

      for (let j = 0; j < numClasses; j++) {
        const score = data[(4 + j) * numDetections + i];
        if (score > maxScore) {
          maxScore = score;
          maxClassId = j;
        }
      }

      const normalizedScore = Math.min(Math.max(maxScore, 0), 1);

      if (normalizedScore > confidenceThreshold && maxClassId < COCO_CLASSES.length) {
        const x1 = Math.max(0, Math.min(1, (xCenter - width / 2) / IMG_SIZE));
        const y1 = Math.max(0, Math.min(1, (yCenter - height / 2) / IMG_SIZE));
        const x2 = Math.max(0, Math.min(1, (xCenter + width / 2) / IMG_SIZE));
        const y2 = Math.max(0, Math.min(1, (yCenter + height / 2) / IMG_SIZE));

        detections.push({
          classId: maxClassId,
          score: normalizedScore,
          bbox: [x1, y1, x2, y2],
          className: COCO_CLASSES[maxClassId] || `Class ${maxClassId}`
        });
      }
    }

    console.log('Found detections:', detections.length);
    return detections;
  };

  // const processDetectionsFromApi = (output: any): Detection[] => {
  //   // Placeholder for processing detections from an API response
  //   return [];
  // }

  const detectFrame = async () => {
    if (!sessionRef.current || !videoRef.current) return;

    try {
      const video = videoRef.current;

      const inputData = preprocessImage(video);

      // @ts-ignore
      const tensor = new ort.Tensor('float32', inputData, [1, 3, IMG_SIZE, IMG_SIZE]);

      const feeds = { images: tensor };
      const results = await sessionRef.current.run(feeds);
      const output = results.output0;

      const dets = processDetections(output);
      setDetections(dets);

      frameCountRef.current++;

    } catch (err) {
      console.error('Detection error:', err);
    }

    if (isDetecting) {
      animationRef.current = requestAnimationFrame(detectFrame);
    }
  };

  const startDetection = () => {
    if (!modelLoaded) {
      setError('Model not loaded yet');
      return;
    }

    setIsDetecting(true);
    setError('');
    frameCountRef.current = 0;

    fpsIntervalRef.current = window.setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      detectFrame();

    }, 1);

  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
    }
    setDetections([]);
    setFps(0);
  };

  const toggleDetection = () => {
    if (isDetecting) {
      stopDetection();
    } else {
      startDetection();
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '2rem',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto'
      }}>
        {/* Header */}
        <div style={{
          textAlign: 'center',
          marginBottom: '2rem',
          color: 'white'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
            <Camera size={40} />
            <h1 style={{ margin: 0, fontSize: '2.5rem', fontWeight: '700' }}>YOLO8 Object Detection</h1>
          </div>
          <p style={{ opacity: 0.9, fontSize: '1.1rem' }}>Real-time AI-powered object recognition</p>
        </div>

        {/* Main Content */}
        <div style={{
          background: 'white',
          borderRadius: '20px',
          padding: '2rem',
          boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
        }}>
          {/* Video Display with Overlay */}
          <div
            ref={containerRef}
            style={{
              position: 'relative',
              marginBottom: '1.5rem',
              borderRadius: '12px',
              overflow: 'hidden',
              background: '#000'
            }}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: '100%',
                display: 'block',
                borderRadius: '12px'
              }}
            />

            {/* Detection Overlay */}
            {isDetecting && detections.map((det, idx) => {
              const [x1, y1, x2, y2] = det.bbox;
              const color = CLASS_COLORS[det.classId % CLASS_COLORS.length];

              return (
                <div
                  key={idx}
                  style={{
                    position: 'absolute',
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${(x2 - x1) * 100}%`,
                    height: `${(y2 - y1) * 100}%`,
                    border: `3px solid ${color}`,
                    boxSizing: 'border-box',
                    pointerEvents: 'none'
                  }}
                >
                  {/* Label */}
                  <div style={{
                    position: 'absolute',
                    top: '-28px',
                    left: '0',
                    background: color,
                    color: 'white',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '14px',
                    fontWeight: '600',
                    whiteSpace: 'nowrap',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
                  }}>
                    {det.className} {(det.score * 100).toFixed(1)}%
                  </div>
                </div>
              );
            })}

            {/* FPS Counter */}
            {isDetecting && (
              <div style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                background: 'rgba(0,0,0,0.7)',
                color: 'white',
                padding: '0.5rem 1rem',
                borderRadius: '8px',
                fontSize: '0.9rem',
                fontWeight: '600'
              }}>
                {fps} FPS
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem'
          }}>
            {/* Confidence Threshold */}
            <div>
              <label style={{
                display: 'block',
                marginBottom: '0.5rem',
                fontWeight: '600',
                color: '#333'
              }}>
                Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={confidenceThreshold * 100}
                onChange={(e) => setConfidenceThreshold(parseInt(e.target.value) / 100)}
                disabled={isDetecting}
                style={{
                  width: '100%',
                  height: '8px',
                  borderRadius: '4px',
                  outline: 'none'
                }}
              />
            </div>

            {/* Action Button */}
            <button
              onClick={toggleDetection}
              disabled={isLoading || !modelLoaded}
              style={{
                padding: '1rem 2rem',
                fontSize: '1.1rem',
                fontWeight: '600',
                color: 'white',
                background: isDetecting ? '#ef4444' : '#10b981',
                border: 'none',
                borderRadius: '12px',
                cursor: isLoading || !modelLoaded ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                transition: 'all 0.3s',
                opacity: isLoading || !modelLoaded ? 0.6 : 1
              }}
              onMouseEnter={(e) => {
                if (!isLoading && modelLoaded) {
                  e.currentTarget.style.transform = 'scale(1.02)';
                  e.currentTarget.style.boxShadow = '0 8px 16px rgba(0,0,0,0.2)';
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              {isLoading ? (
                <>
                  <Loader2 size={24} style={{ animation: 'spin 1s linear infinite' }} />
                  Loading Model...
                </>
              ) : isDetecting ? (
                <>
                  <Pause size={24} />
                  Stop Detection
                </>
              ) : (
                <>
                  <Play size={24} />
                  Start Detection
                </>
              )}
            </button>

            {/* Error Message */}
            {error && (
              <div style={{
                padding: '1rem',
                background: '#fee2e2',
                color: '#dc2626',
                borderRadius: '8px',
                fontSize: '0.9rem'
              }}>
                {error}
              </div>
            )}

            {/* Detections List */}
            {detections.length > 0 && (
              <div style={{
                marginTop: '1rem',
                padding: '1rem',
                background: '#f9fafb',
                borderRadius: '12px',
                maxHeight: '200px',
                overflowY: 'auto'
              }}>
                <h3 style={{
                  margin: '0 0 0.75rem 0',
                  fontSize: '1rem',
                  color: '#333',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  <Square size={20} />
                  Detected Objects ({detections.length})
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {detections.map((det, idx) => (
                    <div
                      key={idx}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '0.5rem',
                        background: 'white',
                        borderRadius: '6px',
                        fontSize: '0.9rem'
                      }}
                    >
                      <span style={{ fontWeight: '600', color: CLASS_COLORS[det.classId % CLASS_COLORS.length] }}>
                        {det.className}
                      </span>
                      <span style={{ color: '#666' }}>
                        {(det.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{
          textAlign: 'center',
          marginTop: '2rem',
          color: 'white',
          opacity: 0.8,
          fontSize: '0.9rem'
        }}>
          Powered by YOLOv8 & ONNX Runtime
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        input[type="range"] {
          -webkit-appearance: none;
          appearance: none;
          background: #e5e7eb;
        }
        
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          background: #667eea;
          cursor: pointer;
          border-radius: 50%;
        }
        
        input[type="range"]::-moz-range-thumb {
          width: 20px;
          height: 20px;
          background: #667eea;
          cursor: pointer;
          border-radius: 50%;
          border: none;
        }
      `}</style>
    </div>
  );
}

export default App;