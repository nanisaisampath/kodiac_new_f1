import React, { useState, useEffect } from 'react';
import { Folder, File, ChevronRight, ChevronDown, ArrowLeft, Filter } from 'lucide-react';
import { api } from '../../services/api';
import { useAppStore } from '../../store/appStore';
import type { S3File } from '../../types';
import { ContextMenu } from './ContextMenu';

interface FileNode extends S3File {
  children?: FileNode[];
  expanded?: boolean;
}

export function FileExplorer() {
  const [files, setFiles] = useState<FileNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState('');
  const [filter, setFilter] = useState('');
  const [extensionFilter, setExtensionFilter] = useState<string>('all');
  
  const { setContextMenu, contextMenu } = useAppStore();

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
  try {
    setLoading(true);
    const fileTree = await api.getS3FileList(); // ← directly get the nested tree now
    setFiles(fileTree);
    setError(null);
  } catch (err) {
    setError(err instanceof Error ? err.message : 'Failed to load files');
  } finally {
    setLoading(false);
  }
  };


  const buildFileTree = (flatFiles: S3File[]): FileNode[] => {
    const tree: FileNode[] = [];
    const pathMap = new Map<string, FileNode>();

    // Sort files: folders first, then by name
    const sortedFiles = flatFiles.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === 'folder' ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    });

    sortedFiles.forEach((file) => {
      const node: FileNode = { ...file, children: file.type === 'folder' ? [] : undefined };
      pathMap.set(file.path, node);

      const parentPath = file.path.substring(0, file.path.lastIndexOf('/'));
      if (parentPath && pathMap.has(parentPath)) {
        const parent = pathMap.get(parentPath)!;
        parent.children?.push(node);
      } else {
        tree.push(node);
      }
    });

    return tree;
  };

  const toggleExpanded = (path: string) => {
    setFiles(prev => updateNodeExpansion(prev, path));
  };

  const updateNodeExpansion = (nodes: FileNode[], targetPath: string): FileNode[] => {
    return nodes.map(node => {
      if (node.path === targetPath) {
        return { ...node, expanded: !node.expanded };
      }
      if (node.children) {
        return { ...node, children: updateNodeExpansion(node.children, targetPath) };
      }
      return node;
    });
  };

  const handleRightClick = (e: React.MouseEvent, filePath: string) => {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      filePath,
    });
  };

  const filterFiles = (nodes: FileNode[]): FileNode[] => {
    return nodes.filter(node => {
      if (node.type === 'folder') {
        const hasMatchingChildren = filterFiles(node.children || []).length > 0;
        const matchesName = node.name.toLowerCase().includes(filter.toLowerCase());
        return matchesName || hasMatchingChildren;
      }
      
      const matchesName = node.name.toLowerCase().includes(filter.toLowerCase());
      const matchesExtension = extensionFilter === 'all' || 
        (node.extension && node.extension.toLowerCase() === extensionFilter.toLowerCase());
      
      return matchesName && matchesExtension;
    }).map(node => ({
      ...node,
      children: node.children ? filterFiles(node.children) : undefined
    }));
  };

  const renderFileNode = (node: FileNode, level = 0) => {
    const isFolder = node.type === 'folder';
    const isDicomFile = node.extension && ['dcm', 'dicom', 'e2e', 'fds', 'fda'].includes(node.extension.toLowerCase());
    
    return (
      <div key={node.path}>
        <div
          className={`flex items-center py-1 px-2 hover:bg-blue-50 cursor-pointer select-none ${
            level > 0 ? `ml-${level * 4}` : ''
          }`}
          onClick={() => isFolder && toggleExpanded(node.path)}
          onContextMenu={isDicomFile ? (e) => handleRightClick(e, node.path) : undefined}
          style={{ paddingLeft: `${level * 16 + 8}px` }}
        >
          {isFolder && (
            <span className="mr-1">
              {node.expanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </span>
          )}
          
          <span className="mr-2">
            {isFolder ? (
              <Folder className="w-4 h-4 text-blue-500" />
            ) : (
              <File className={`w-4 h-4 ${isDicomFile ? 'text-green-500' : 'text-gray-500'}`} />
            )}
          </span>
          
          <span className="text-sm truncate flex-1">{node.name}</span>
          
          {node.size && (
            <span className="text-xs text-gray-500 ml-2">
              {(node.size / 1024 / 1024).toFixed(1)}MB
            </span>
          )}
        </div>
        
        {isFolder && node.expanded && node.children && (
          <div>
            {node.children.map(child => renderFileNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  const breadcrumbs = currentPath.split('/').filter(Boolean);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-red-600">
        <p>Error loading files: {error}</p>
        <button
          onClick={loadFiles}
          className="mt-2 px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    );
  }

  const filteredFiles = filterFiles(files);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-3 border-b border-gray-200">
        <div className="flex items-center gap-2 mb-2">
          {breadcrumbs.length > 0 && (
            <button
              onClick={() => setCurrentPath('')}
              className="p-1 hover:bg-gray-100 rounded"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
          )}
          <h3 className="font-semibold text-gray-800">Files</h3>
        </div>
        
        {/* Breadcrumbs */}
        {breadcrumbs.length > 0 && (
          <div className="text-xs text-gray-600 mb-2">
            {breadcrumbs.join(' / ')}
          </div>
        )}
        
        {/* Filters */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Filter files..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded"
            />
          </div>
          
          <select
            value={extensionFilter}
            onChange={(e) => setExtensionFilter(e.target.value)}
            className="w-full px-2 py-1 text-xs border border-gray-300 rounded"
          >
            <option value="all">All files</option>
            <option value="dcm">DICOM (.dcm)</option>
            <option value="e2e">E2E (.e2e)</option>
            <option value="fds">FDS (.fds)</option>
            <option value="fda">FDA (.fda)</option>
          </select>
        </div>
      </div>
      
      {/* File Tree */}
      <div className="flex-1 overflow-y-auto">
        {filteredFiles.length === 0 ? (
          <div className="p-4 text-gray-500 text-sm">
            No files found matching your criteria.
          </div>
        ) : (
          filteredFiles.map(node => renderFileNode(node))
        )}
      </div>
      
      {contextMenu && <ContextMenu />}
    </div>
  );
}