const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  const apiProxy = createProxyMiddleware({
    target: 'http://localhost:5000',
    changeOrigin: true,
    secure: false,
    logLevel: 'debug',
    onError: (err, req, res) => {
      console.error('Proxy Error:', err);
      res.status(500).json({ 
        error: 'Proxy Error', 
        message: err.message,
        details: 'Failed to connect to backend server. Please ensure the Flask server is running on port 5000.'
      });
    },
    pathRewrite: {
      '^/api': '/api'
    },
    onProxyRes: (proxyRes, req, res) => {
      console.log('Proxy Response:', {
        status: proxyRes.statusCode,
        path: req.path,
        method: req.method
      });
    }
  });

  app.use('/api', apiProxy);
};
