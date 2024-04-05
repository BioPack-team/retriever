var path = require('path');
var fs = require('fs');
var webpack = require('webpack');

module.exports = {
  entry: './src/index.ts',
  mode: 'production',
  output: {
    path: path.resolve(__dirname),
    filename: 'bundle.js',
    library: 'MetaKG',
    libraryExport: 'default',
    libraryTarget: 'umd',
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js'],
    fallback: {
      fs: false,
      path: require.resolve('path-browserify')
    }
  },
  module: {
    rules: [
      {
        test: /\.(ts|js)x?$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [['@babel/env', {
              useBuiltIns: 'entry',
              corejs: '3'
            }], 
            '@babel/preset-typescript'],
            plugins: [
                '@babel/proposal-class-properties', 
                '@babel/proposal-object-rest-spread',
                '@babel/plugin-transform-runtime'
            ]
          },
        }
      },
    ],
  },
  plugins: [],
};
