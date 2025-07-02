import type { S3FileStructure, DicomFrame, DicomMetadata, CacheStatus, S3Credentials } from '../types';

const API_BASE = '/api';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorText = await response.text();
    console.error(`API Error ${response.status}:`, errorText);
    throw new ApiError(response.status, `API Error: ${response.statusText} - ${errorText}`);
  }
  return response.json();
}

export const api = {
  // S3 and File Operations
  async getS3FileList(): Promise<any[]> {
  console.log('🌐 Calling /api/s3-flat-list...');
  const response = await fetch(`${API_BASE}/s3-flat-list`);

  if (!response.ok) {
    const errorText = await response.text();
    console.error('❌ S3 file list error:', errorText);
    throw new ApiError(response.status, `Failed to load S3 files: ${response.statusText}`);
  }

  const data = await response.json();
  console.log('📦 S3 file list response:', {
    type: typeof data,
    isArray: Array.isArray(data),
    length: Array.isArray(data) ? data.length : 'N/A'
  });

  if (!Array.isArray(data)) {
    console.warn('⚠️ S3 list is not an array. Returning as-is.');
    return data;
  }

  // Step 1: Transform all files, safely checking .key
  const flatFiles = data
    .filter(file => typeof file.key === 'string') // skip invalid entries
    .map(file => ({
      name: file.key.split('/').pop() || file.key,
      path: file.key,
      type: 'file' as const,
      size: file.size ?? 0,
      modified: file.last_modified ?? null,
      extension: file.key.includes('.') ? file.key.split('.').pop()?.toLowerCase() : ''
    }));

  // Step 2: Build nested tree structure
  const root: any[] = [];

  for (const file of flatFiles) {
    const parts = file.path.split('/');
    let currentLevel = root;

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const currentPath = parts.slice(0, i + 1).join('/');

      if (!part) continue; // skip empty

      if (i === parts.length - 1) {
        // Last part is the file
        currentLevel.push({ ...file });
      } else {
        // Folder
        let folder = currentLevel.find(
          (f: any) => f.type === 'folder' && f.name === part
        );
        if (!folder) {
          folder = {
            name: part,
            path: currentPath,
            type: 'folder' as const,
            children: []
          };
          currentLevel.push(folder);
        }
        currentLevel = folder.children;
      }
    }
  }

  console.log('🌳 Nested folder tree:', root);
  return root;
  },
  

  async getFileCrc(path: string): Promise<{ crc: string }> {
    console.log('🔍 Getting CRC for:', path);
    const response = await fetch(`${API_BASE}/get-file-crc?path=${encodeURIComponent(path)}`);
    const result = await handleResponse<{ crc: string }>(response);
    console.log('✅ CRC result:', result);
    return result;
  },

  async getCacheStatus(): Promise<CacheStatus> {
    console.log('📊 Getting cache status...');
    const response = await fetch(`${API_BASE}/cache-status`);
    const status = await handleResponse<any>(response);
    console.log('📊 Cache status response:', status);
    
    // Transform backend response to match frontend expectations
    return {
      total_files: status.memory_entries || 0,
      cached_files: status.disk_entries || 0,
      cache_size_mb: status.total_size_mb || 0,
      hit_rate: status.disk_entries > 0 ? status.disk_entries / (status.memory_entries + status.disk_entries) : 0
    };
  },

  // DICOM Operations
  async getDicomFrame(filePath: string, frame: number, crc?: string): Promise<Blob> {
    console.log('🖼️ Getting DICOM frame:', { filePath, frame, crc });
    
    // First, download/process the file from S3 if needed
    await this.downloadDicomFromS3(filePath);
    
    const url = new URL(`${API_BASE}/view_dicom_png`, window.location.origin);
    url.searchParams.set('dicom_file_path', filePath);
    url.searchParams.set('frame', frame.toString());
    if (crc) {
      url.searchParams.set('v', crc);
    }
    
    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new ApiError(response.status, `Failed to load DICOM frame: ${response.statusText}`);
    }
    return response.blob();
  },

  async getDicomFrames(filePath: string): Promise<{ frames: number[] }> {
    console.log('📋 Getting DICOM frames list for:', filePath);
    
    // First ensure the file is processed
    const processResponse = await this.downloadDicomFromS3(filePath);
    
    // Extract the dicom_file_path from the response
    const fileKey = processResponse.dicom_file_path;
    
    const response = await fetch(`${API_BASE}/view_frames/${encodeURIComponent(fileKey)}`);
    const frameData = await handleResponse<{ number_of_frames: number; frame_urls: string[] }>(response);
    
    // Generate frame indices based on number of frames
    const frames = Array.from({ length: frameData.number_of_frames }, (_, i) => i);
    console.log('📋 Frames result:', { total: frameData.number_of_frames, frames });
    return { frames };
  },

  async getFlattenedDicom(filePath: string): Promise<Blob> {
    console.log('🔄 Getting flattened DICOM for:', filePath);
    
    // First ensure the file is processed
    const processResponse = await this.downloadDicomFromS3(filePath);
    const fileKey = processResponse.dicom_file_path;
    
    const url = new URL(`${API_BASE}/flatten_dicom_image`, window.location.origin);
    url.searchParams.set('dicom_file_path', fileKey);
    
    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new ApiError(response.status, `Failed to load flattened DICOM: ${response.statusText}`);
    }
    return response.blob();
  },

  async checkDicomReady(filePath: string): Promise<{ ready: boolean }> {
    const response = await fetch(`${API_BASE}/check_dicom_ready?dicom_file_path=${encodeURIComponent(filePath)}`);
    return handleResponse<{ ready: boolean }>(response);
  },

  // S3 File Download and Processing
  async downloadDicomFromS3(path: string): Promise<{ 
    message: string; 
    number_of_frames: number; 
    dicom_file_path: string;
    cache_source: string;
  }> {
    console.log('⬇️ Downloading/processing from S3:', path);
    const response = await fetch(`${API_BASE}/download_dicom_from_s3?path=${encodeURIComponent(path)}`);
    const result = await handleResponse<{ 
      message: string; 
      number_of_frames: number; 
      dicom_file_path: string;
      cache_source: string;
    }>(response);
    console.log('✅ S3 download result:', result);
    return result;
  },

  // File Processing
  async processUpload(file: File): Promise<{ success: boolean; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE}/process-upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<{ success: boolean; message: string }>(response);
  },

  // S3 Credentials
  async setS3Credentials(credentials: S3Credentials): Promise<{ success: boolean }> {
    console.log('🔐 Setting S3 credentials...');
    const response = await fetch(`${API_BASE}/set-s3-credentials`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        accessKey: credentials.accessKey,
        secretKey: credentials.secretKey,
        region: credentials.region,
        bucket: credentials.bucketName,
        saveToEnv: true
      }),
    });
    
    const result = await handleResponse<{ message: string }>(response);
    console.log('✅ S3 credentials set:', result);
    return { success: true };
  },

  // S3 Status Check
  async getS3Status(): Promise<{ configured: boolean; needs_credentials: boolean; message: string }> {
    console.log('🔍 Checking S3 status...');
    const response = await fetch(`${API_BASE}/s3-status`);
    const result = await handleResponse<{ configured: boolean; needs_credentials: boolean; message: string }>(response);
    console.log('📊 S3 status result:', result);
    return result;
  },

  // DICOM Support Status
  async getDicomSupportStatus(): Promise<{
    pylibjpeg_available: boolean;
    gdcm_available: boolean;
    opencv_available: boolean;
    supported_compressions: string[];
  }> {
    const response = await fetch(`${API_BASE}/dicom_support_status`);
    return handleResponse<{
      pylibjpeg_available: boolean;
      gdcm_available: boolean;
      opencv_available: boolean;
      supported_compressions: string[];
    }>(response);
  }
};