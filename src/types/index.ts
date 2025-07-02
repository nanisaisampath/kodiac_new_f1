export interface S3File {
  name: string;
  path: string;
  type: 'file' | 'folder';
  size?: number;
  modified?: string;
  extension?: string;
}

export interface S3FileStructure {
  files: S3File[];
  folders: S3File[];
}

export interface DicomFrame {
  index: number;
  url: string;
  cached?: boolean;
}

export interface DicomMetadata {
  frameCount: number;
  width: number;
  height: number;
  imageType: string;
  studyDate?: string;
  patientName?: string;
  modality?: string;
}

export interface ViewportState {
  id: number;
  filePath: string | null;
  frames: DicomFrame[];
  currentFrame: number;
  zoom: number;
  loading: boolean;
  error: string | null;
  metadata: DicomMetadata | null;
  crcKey: string | null;
  loadTime: number | null;
  // Backend-specific fields
  processedFileKey?: string | null;
  isOct?: boolean;
}

export interface CacheStatus {
  total_files: number;
  cached_files: number;
  cache_size_mb: number;
  hit_rate: number;
}

export interface S3Credentials {
  accessKey: string;
  secretKey: string;
  region: string;
  bucketName: string;
}

export type LayoutMode = 'side-by-side' | 'stacked';

export interface AppSettings {
  layoutMode: LayoutMode;
  bindSliders: boolean;
  sidebarExpanded: boolean;
}

// Backend-specific types
export interface BackendFileProcessResponse {
  message: string;
  number_of_frames: number;
  dicom_file_path: string;
  cache_source: string;
  compression_info?: {
    is_compressed: boolean;
    compression_type: string;
  };
}

export interface BackendCacheStatus {
  memory_entries: number;
  disk_cache_size: number;
  disk_entries: number;
  total_size_mb: number;
  crc_mappings: number;
}

export interface BackendS3Status {
  configured: boolean;
  needs_credentials: boolean;
  bucket?: string;
  message: string;
  error?: string;
}