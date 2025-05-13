# Vercel Deployment Guide

This document explains how to deploy the Face Blurring web application on Vercel.

## Prerequisites

1. A [Vercel account](https://vercel.com/signup)
2. [Git](https://git-scm.com/downloads) installed
3. Your code pushed to a GitHub, GitLab, or Bitbucket repository

## Deployment Steps

### 1. Prepare Your Code for Deployment

Your repository should include:

- `api/index.py` - The serverless function entry point
- `vercel.json` - Configuration file for Vercel
- `templates/` directory with HTML templates
- `static/` directory with CSS files
- `haarcascade_frontalface_alt.xml` - The face detection model
- `requirements.txt` - Python dependencies

### 2. Deploy to Vercel

#### Using the Vercel Dashboard

1. Log in to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your Git repository
4. Select the repository containing your Face Blurring application
5. Configure your project:
   - Framework Preset: Other
   - Build Command: Leave empty
   - Output Directory: Leave empty
   - Install Command: `pip install -r requirements.txt`
6. Click "Deploy"

#### Using the Vercel CLI

1. Install Vercel CLI:

   ```
   npm install -g vercel
   ```

2. Log in to Vercel:

   ```
   vercel login
   ```

3. Navigate to your project directory and deploy:

   ```
   cd your-project-directory
   vercel
   ```

4. Follow the prompts to configure your deployment

### 3. Environment Variables

If needed, set environment variables through the Vercel dashboard:

1. Go to your project in Vercel dashboard
2. Go to Settings > Environment Variables
3. Add any required variables

### 4. Troubleshooting

- **Deployment Errors**: Check the build logs in Vercel dashboard
- **Runtime Errors**: Check the Function Logs in Vercel dashboard
- **Missing Files**: Ensure all required files are included in your repository

### 5. Custom Domain (Optional)

1. Go to your project in Vercel dashboard
2. Go to Settings > Domains
3. Add your custom domain and follow the verification process

## Limitations

1. Vercel has a function execution time limit (currently 10 seconds for hobby plans)
2. There's a maximum deployment size limit
3. `/tmp` directory is writable but ephemeral (files disappear between invocations)

For more details, check the [Vercel documentation](https://vercel.com/docs/frameworks/python).
