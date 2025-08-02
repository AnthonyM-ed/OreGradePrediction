# 🚀 Ore Grade Prediction System - Windows Startup Script
# This script starts both backend and frontend automatically

Write-Host "🚀 STARTING ORE GRADE PREDICTION SYSTEM" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""

# Get the current directory (should be project root)
$projectRoot = Get-Location
Write-Host "📁 Project Root: $projectRoot" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "backend\manage.py") -or -not (Test-Path "client\package.json")) {
    Write-Host "❌ Error: Please run this script from the project root directory" -ForegroundColor Red
    Write-Host "   The directory should contain both 'backend' and 'client' folders" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Project structure validated" -ForegroundColor Green
Write-Host ""

# Function to start backend
function Start-Backend {
    Write-Host "🔧 Starting Django Backend..." -ForegroundColor Yellow
    
    # Change to backend directory
    Set-Location "$projectRoot\backend"
    
    # Start Django server in a new PowerShell window
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command", 
        "Write-Host '🔧 DJANGO BACKEND SERVER' -ForegroundColor Green; Write-Host 'URL: http://127.0.0.1:8000' -ForegroundColor Cyan; Write-Host ''; python manage.py runserver"
    ) -WindowStyle Normal
    
    Write-Host "✅ Backend starting in new window..." -ForegroundColor Green
    Write-Host "   URL: http://127.0.0.1:8000" -ForegroundColor Cyan
}

# Function to start frontend
function Start-Frontend {
    Write-Host "⚛️  Starting React Frontend..." -ForegroundColor Yellow
    
    # Change to client directory
    Set-Location "$projectRoot\client"
    
    # Check if node_modules exists
    if (-not (Test-Path "node_modules")) {
        Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
        npm install
    }
    
    # Start React dev server in a new PowerShell window
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Write-Host '⚛️  REACT FRONTEND SERVER' -ForegroundColor Green; Write-Host 'URL: http://localhost:5173' -ForegroundColor Cyan; Write-Host ''; npm run dev"
    ) -WindowStyle Normal
    
    Write-Host "✅ Frontend starting in new window..." -ForegroundColor Green
    Write-Host "   URL: http://localhost:5173" -ForegroundColor Cyan
}

# Function to test system
function Test-System {
    Write-Host "🧪 Would you like to test the API? (y/n): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "⏳ Waiting 10 seconds for servers to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        Set-Location "$projectRoot\backend"
        Write-Host "🔬 Running API tests..." -ForegroundColor Yellow
        python test_prediction_api.py
    }
}

# Main execution
try {
    # Start backend
    Start-Backend
    Start-Sleep -Seconds 2
    
    # Start frontend
    Start-Frontend
    Start-Sleep -Seconds 2
    
    # Return to project root
    Set-Location $projectRoot
    
    Write-Host ""
    Write-Host "🎉 SYSTEM STARTUP COMPLETE!" -ForegroundColor Green
    Write-Host "=========================" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 System URLs:" -ForegroundColor Cyan
    Write-Host "   Frontend: http://localhost:5173" -ForegroundColor White
    Write-Host "   Backend:  http://127.0.0.1:8000" -ForegroundColor White
    Write-Host ""
    Write-Host "📝 How to use:" -ForegroundColor Cyan
    Write-Host "   1. Open http://localhost:5173 in your browser" -ForegroundColor White
    Write-Host "   2. Click 'AI Prediction' in the sidebar" -ForegroundColor White
    Write-Host "   3. Fill the form and click 'Predict Grade'" -ForegroundColor White
    Write-Host ""
    Write-Host "🛠️  To stop servers: Close the PowerShell windows" -ForegroundColor Cyan
    Write-Host ""
    
    # Optional testing
    Test-System
    
} catch {
    Write-Host "❌ Error starting system: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Try running the commands manually:" -ForegroundColor Yellow
    Write-Host "   1. cd backend; python manage.py runserver" -ForegroundColor White
    Write-Host "   2. cd client; npm run dev" -ForegroundColor White
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
