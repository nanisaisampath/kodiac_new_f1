import React, { useState } from 'react';
import { X, Save, Eye, EyeOff } from 'lucide-react';
import { useAppStore } from '../../store/appStore';
import { api } from '../../services/api';
import type { S3Credentials } from '../../types';

export function S3CredentialsModal() {
  const { showS3Modal, setShowS3Modal } = useAppStore();
  const [credentials, setCredentials] = useState<S3Credentials>({
    accessKey: '',
    secretKey: '',
    region: 'us-east-1',
    bucketName: '',
  });
  const [showSecretKey, setShowSecretKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  if (!showS3Modal) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!credentials.accessKey || !credentials.secretKey || !credentials.bucketName) {
      setError('Please fill in all required fields');
      return;
    }

    try {
      setSaving(true);
      setError(null);
      
      await api.setS3Credentials(credentials);
      
      setSuccess(true);
      setTimeout(() => {
        setShowS3Modal(false);
        setSuccess(false);
        setCredentials({
          accessKey: '',
          secretKey: '',
          region: 'us-east-1',
          bucketName: '',
        });
      }, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save credentials');
    } finally {
      setSaving(false);
    }
  };

  const handleClose = () => {
    setShowS3Modal(false);
    setError(null);
    setSuccess(false);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-800">S3 Credentials</h2>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          {success && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-md">
              <p className="text-sm text-green-600">Credentials saved successfully!</p>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Access Key ID *
            </label>
            <input
              type="text"
              value={credentials.accessKey}
              onChange={(e) => setCredentials({ ...credentials, accessKey: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="AKIAIOSFODNN7EXAMPLE"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Secret Access Key *
            </label>
            <div className="relative">
              <input
                type={showSecretKey ? 'text' : 'password'}
                value={credentials.secretKey}
                onChange={(e) => setCredentials({ ...credentials, secretKey: e.target.value })}
                className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
                required
              />
              <button
                type="button"
                onClick={() => setShowSecretKey(!showSecretKey)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                {showSecretKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Region
            </label>
            <select
              value={credentials.region}
              onChange={(e) => setCredentials({ ...credentials, region: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="us-east-1">US East (N. Virginia)</option>
              <option value="us-east-2">US East (Ohio)</option>
              <option value="us-west-1">US West (N. California)</option>
              <option value="us-west-2">US West (Oregon)</option>
              <option value="eu-west-1">Europe (Ireland)</option>
              <option value="eu-central-1">Europe (Frankfurt)</option>
              <option value="ap-southeast-1">Asia Pacific (Singapore)</option>
              <option value="ap-northeast-1">Asia Pacific (Tokyo)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Bucket Name *
            </label>
            <input
              type="text"
              value={credentials.bucketName}
              onChange={(e) => setCredentials({ ...credentials, bucketName: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="my-dicom-bucket"
              required
            />
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 text-gray-600 hover:text-gray-800 font-medium"
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {saving ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4" />
                  Save Credentials
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}