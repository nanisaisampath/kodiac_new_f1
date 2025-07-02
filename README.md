# DICOM Viewer - Medical Imaging Platform

A comprehensive React frontend for viewing DICOM and OCT medical images with AWS S3 integration, built to work with a FastAPI backend.

## 🚀 Features

### 📁 File Management
- **S3 File Browser**: Browse files stored in AWS S3 with folder tree navigation
- **Smart Filtering**: Filter by file extension and search by name
- **Context Menu**: Right-click DICOM files to load in either viewport
- **Breadcrumb Navigation**: Easy navigation through folder structure

### 🖼️ DICOM Viewing
- **Dual Viewports**: View two DICOM images simultaneously
- **Multi-frame Support**: Navigate through DICOM frames with interactive sliders
- **OCT Processing**: Automatic flattened OCT image loading when available
- **Zoom Controls**: Zoom in/out and reset zoom functionality
- **Performance Monitoring**: Track image load times

### 🔄 Layout & Synchronization
- **Flexible Layouts**: Switch between side-by-side and stacked viewport arrangements
- **Frame Synchronization**: Link frame sliders across both viewports
- **Responsive Design**: Optimized for various screen sizes

### ⚡ Performance & Caching
- **Intelligent Caching**: CRC-based image caching to prevent redundant downloads
- **Cache Metrics**: Real-time cache status monitoring
- **Background Loading**: Smooth loading experience with progress indicators

### 🛠️ Tools & Integration
- **File Upload**: E2E to DICOM conversion via drag-and-drop
- **S3 Credentials**: Secure credential management with encrypted storage
- **Metadata Display**: View DICOM metadata and image properties
- **Collapsible Sidebar**: Space-efficient tool access

## 🏗️ Architecture

### Frontend Stack
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for responsive styling
- **Zustand** for state management
- **Lucide React** for icons

### Backend Integration
The frontend connects to a FastAPI backend with these key endpoints:
- `/api/s3-flat-list` - File structure retrieval
- `/api/view_dicom_png` - DICOM frame rendering
- `/api/flatten_dicom_image` - OCT processing
- `/api/get-file-crc` - Caching validation
- `/api/cache-status` - Performance metrics

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- FastAPI backend running on localhost:8000

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```
   This creates a `static/` directory that your FastAPI backend can serve.

### Backend Integration

Your FastAPI backend should serve the built frontend:

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the React build
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

## 🔧 Configuration

### Vite Proxy
The development server automatically proxies `/api/*` requests to `localhost:8000`. This is configured in `vite.config.ts`.

### Environment Variables
The app uses these environment variables (handled by the backend):
- S3 access credentials
- Cache configuration
- API endpoints

## 📁 Project Structure

```
src/
├── components/           # React components
│   ├── DicomViewer/     # DICOM viewing components
│   ├── FileExplorer/    # S3 file browser
│   ├── Layout/          # Layout controls
│   ├── Modals/          # Modal dialogs
│   └── Sidebar/         # Navigation sidebar
├── services/            # API integration
├── store/               # Zustand state management
├── types/               # TypeScript definitions
└── App.tsx              # Main application component
```

## 🎨 Design System

### Colors
- **Primary**: Blue (#2563EB) - Navigation and primary actions
- **Secondary**: Teal (#0D9488) - Secondary elements
- **Accent**: Orange (#EA580C) - Highlights and warnings
- **Success**: Green - Successful operations
- **Error**: Red - Error states

### Typography
- Clean, medical-grade typography optimized for technical data
- Consistent spacing using 8px grid system
- High contrast ratios for accessibility

### Responsive Breakpoints
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

## 🧪 Development

### Code Quality
- ESLint for code linting
- TypeScript for type safety
- Component-based architecture
- Proper separation of concerns

### Performance Considerations
- Image caching with CRC validation
- Lazy loading of DICOM frames
- Optimized re-renders with Zustand
- Efficient file tree rendering

## 🔒 Security

- Secure S3 credential handling
- No sensitive data in client-side storage
- API proxy to hide backend endpoints
- Input validation and sanitization

## 📊 Monitoring

- Real-time cache metrics
- Load time tracking
- Error boundary handling
- Performance monitoring dashboard

## 🤝 Contributing

1. Follow the existing code style
2. Add TypeScript types for new features
3. Ensure responsive design
4. Test with various DICOM files
5. Update documentation

## 📄 License

This project is part of a medical imaging system. Please ensure compliance with relevant medical device regulations and HIPAA requirements when handling patient data.