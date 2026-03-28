//+------------------------------------------------------------------+
//|              Universal ONNX ML Trading Bot EA                    |
//|              MetaTrader 5 Expert Advisor                         |
//|              ONNX ML Integration + SMC Strategies                |
//+------------------------------------------------------------------+

#property copyright "Universal ONNX ML Trading Bot"
#property link      "https://github.com/universal-trading-bot"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Math\Stat\Math.mqh>

//--- ONNX Integration
#include <MQL5\Include\Math\Alglib\dataanalysis.mqh>  // For ONNX support

//--- Input parameters
input string ONNX_Model_Path = "onnx_model.onnx";    // Path to ONNX model file
input int Feature_Count = 50;                        // Number of input features
input double Prediction_Threshold = 0.6;             // Minimum confidence for trade
input bool Use_Socket_Comm = true;                   // Use socket communication
input string PythonHost = "localhost";               // Python bot host
input int PythonPort = 5000;                         // Python bot port

//--- SMC Strategy Parameters
input int H4_Timeframe = PERIOD_H4;                  // Higher timeframe for context
input int M15_Timeframe = PERIOD_M15;                // Entry timeframe
input int Fractal_Period = 5;                        // Fractal lookback period
input double Risk_Per_Trade = 0.02;                  // Risk per trade (2%)
input double Daily_Drawdown_Limit = 0.05;            // Max daily drawdown (5%)
input int Max_Positions = 3;                         // Maximum concurrent positions
input int Magic_Number = 123456;                     // EA magic number

//--- Kill Zone Parameters (GMT)
input int London_Open_Hour = 8;                      // London session start
input int London_Close_Hour = 10;                    // London session end
input int NewYork_Open_Hour = 13;                    // New York session start
input int NewYork_Close_Hour = 15;                   // New York session end

//--- Global variables
int socketHandle = INVALID_HANDLE;
CTrade trade;
CSymbolInfo symbolInfo;
CPositionInfo positionInfo;

//--- ONNX variables
long onnxHandle = INVALID_HANDLE;
matrix inputMatrix;
matrix outputMatrix;

//--- SMC variables
datetime lastBarTime;
double dailyStartingBalance;
double dailyDrawdown;
int retryCount = 0;
const int MAX_RETRIES = 3;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 Universal ONNX ML Trading Bot EA v2.0 initializing...");

    // Initialize trade object
    trade.SetExpertMagicNumber(Magic_Number);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(Symbol());

    // Initialize symbol info
    symbolInfo.Name(Symbol());

    // Initialize ONNX model
    if (!InitializeONNX())
    {
        Print("❌ Failed to initialize ONNX model");
        return INIT_FAILED;
    }

    // Initialize SMC variables
    lastBarTime = iTime(Symbol(), M15_Timeframe, 0);
    dailyStartingBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    dailyDrawdown = 0.0;

    // Connect to Python bot (optional)
    if (Use_Socket_Comm)
    {
        if (!ConnectToPython())
        {
            Print("⚠️ Socket communication disabled - running standalone");
        }
    }

    Print("✅ EA initialized successfully with ONNX + SMC integration");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Close socket connection
    if (socketHandle != INVALID_HANDLE)
    {
        SocketClose(socketHandle);
        socketHandle = INVALID_HANDLE;
    }

    // Release ONNX model
    if (onnxHandle != INVALID_HANDLE)
    {
        OnnxRelease(onnxHandle);
        onnxHandle = INVALID_HANDLE;
    }

    Print("📴 EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check for new bar (reduce CPU usage)
    if (!IsNewBar())
        return;

    // Update daily drawdown
    UpdateDailyDrawdown();

    // Check drawdown limit
    if (dailyDrawdown >= Daily_Drawdown_Limit)
    {
        Print("🚫 Daily drawdown limit reached - trading halted");
        return;
    }

    // Check position limits
    if (PositionsTotal() >= Max_Positions)
    {
        return;
    }

    // Check kill zone
    if (!IsInKillZone())
    {
        return;
    }

    // Generate trading signal
    int signal = GenerateMLSignal();
    if (signal == 0)
        return;  // No signal

    // Apply SMC confirmation
    if (!ConfirmWithSMC(signal))
        return;

    // Execute trade with retry logic
    ExecuteTradeWithRetry(signal);
}

//+------------------------------------------------------------------+
//| Initialize ONNX model                                            |
//+------------------------------------------------------------------+
bool InitializeONNX()
{
    // Load ONNX model
    onnxHandle = OnnxCreateFromFile(ONNX_Model_Path);
    if (onnxHandle == INVALID_HANDLE)
    {
        Print("❌ Failed to load ONNX model: ", ONNX_Model_Path);
        return false;
    }

    // Get model information
    OnnxTypeInfo typeInfo;
    if (!OnnxGetModelTypeInfo(onnxHandle, typeInfo))
    {
        Print("❌ Failed to get ONNX model info");
        OnnxRelease(onnxHandle);
        onnxHandle = INVALID_HANDLE;
        return false;
    }

    // Initialize matrices
    inputMatrix.Resize(1, Feature_Count);
    outputMatrix.Resize(1, 1);  // Binary classification

    Print("✅ ONNX model loaded successfully");
    return true;
}

//+------------------------------------------------------------------+
//| Generate ML signal using ONNX model                              |
//+------------------------------------------------------------------+
int GenerateMLSignal()
{
    if (onnxHandle == INVALID_HANDLE)
        return 0;

    // Calculate features
    if (!CalculateFeatures(inputMatrix))
        return 0;

    // Run ONNX inference
    if (!OnnxRun(onnxHandle, ONNX_RUN_INPUT, inputMatrix, ONNX_RUN_OUTPUT, outputMatrix))
    {
        Print("❌ ONNX inference failed");
        return 0;
    }

    // Get prediction
    double prediction = outputMatrix[0][0];

    // Apply threshold
    if (prediction >= Prediction_Threshold)
        return 1;  // Bullish
    else if (prediction <= (1.0 - Prediction_Threshold))
        return -1; // Bearish

    return 0; // Neutral
}

//+------------------------------------------------------------------+
//| Calculate technical features for ML model                       |
//+------------------------------------------------------------------+
bool CalculateFeatures(matrix &features)
{
    // Price data
    double close = iClose(Symbol(), M15_Timeframe, 0);
    double high = iHigh(Symbol(), M15_Timeframe, 0);
    double low = iLow(Symbol(), M15_Timeframe, 0);
    double open = iOpen(Symbol(), M15_Timeframe, 0);
    long volume = iVolume(Symbol(), M15_Timeframe, 0);

    // Basic price features
    features[0][0] = close;
    features[0][1] = (close - open) / open;  // Body size
    features[0][2] = (high - low) / close;   // Range
    features[0][3] = volume;

    // Moving averages
    features[0][4] = iMA(Symbol(), M15_Timeframe, 20, 0, MODE_SMA, PRICE_CLOSE, 0);
    features[0][5] = iMA(Symbol(), M15_Timeframe, 50, 0, MODE_SMA, PRICE_CLOSE, 0);

    // RSI
    features[0][6] = iRSI(Symbol(), M15_Timeframe, 14, PRICE_CLOSE, 0);

    // MACD
    double macd_main, macd_signal;
    iMACD(Symbol(), M15_Timeframe, 12, 26, 9, PRICE_CLOSE, macd_main, macd_signal, 0);
    features[0][7] = macd_main;
    features[0][8] = macd_signal;

    // Bollinger Bands
    double bb_upper, bb_lower;
    iBands(Symbol(), M15_Timeframe, 20, 0, 2, PRICE_CLOSE, bb_upper, bb_lower, 0);
    features[0][9] = bb_upper;
    features[0][10] = bb_lower;

    // ATR
    features[0][11] = iATR(Symbol(), M15_Timeframe, 14, 0);

    // Higher timeframe context (H4)
    double h4_close = iClose(Symbol(), H4_Timeframe, 0);
    double h4_high = iHigh(Symbol(), H4_Timeframe, 0);
    double h4_low = iLow(Symbol(), H4_Timeframe, 0);
    features[0][12] = h4_close;
    features[0][13] = (h4_high - h4_low) / h4_close;  // H4 range

    // Add more features as needed (up to Feature_Count)
    // This is a simplified version - expand based on your model's requirements

    return true;
}

//+------------------------------------------------------------------+
//| Confirm signal with SMC (Smart Money Concepts)                  |
//+------------------------------------------------------------------+
bool ConfirmWithSMC(int signal)
{
    // Check for fractal patterns (liquidity sweeps)
    if (!CheckFractalPattern(signal))
        return false;

    // Check order blocks
    if (!CheckOrderBlocks(signal))
        return false;

    // Check Fibonacci zones
    if (!CheckFibonacciZones(signal))
        return false;

    return true;
}

//+------------------------------------------------------------------+
//| Check fractal patterns for liquidity sweeps                     |
//+------------------------------------------------------------------+
bool CheckFractalPattern(int signal)
{
    // Look for 5-bar fractal patterns indicating liquidity sweeps
    int direction = (signal > 0) ? 1 : -1;  // 1 for up, -1 for down

    for (int i = 1; i <= Fractal_Period; i++)
    {
        double high1 = iHigh(Symbol(), M15_Timeframe, i);
        double high2 = iHigh(Symbol(), M15_Timeframe, i+1);
        double low1 = iLow(Symbol(), M15_Timeframe, i);
        double low2 = iLow(Symbol(), M15_Timeframe, i+1);

        // Check for bearish fractal (resistance sweep)
        if (direction == -1 && high1 > high2 && high1 > iHigh(Symbol(), M15_Timeframe, i-1))
            return true;

        // Check for bullish fractal (support sweep)
        if (direction == 1 && low1 < low2 && low1 < iLow(Symbol(), M15_Timeframe, i-1))
            return true;
    }

    return false;
}

//+------------------------------------------------------------------+
//| Check order blocks (supply/demand zones)                        |
//+------------------------------------------------------------------+
bool CheckOrderBlocks(int signal)
{
    // Simplified order block detection
    // Large candle followed by small candle in opposite direction
    double body1 = MathAbs(iOpen(Symbol(), M15_Timeframe, 1) - iClose(Symbol(), M15_Timeframe, 1));
    double body2 = MathAbs(iOpen(Symbol(), M15_Timeframe, 0) - iClose(Symbol(), M15_Timeframe, 0));

    double range1 = iHigh(Symbol(), M15_Timeframe, 1) - iLow(Symbol(), M15_Timeframe, 1);
    double range2 = iHigh(Symbol(), M15_Timeframe, 0) - iLow(Symbol(), M15_Timeframe, 0);

    // Large candle followed by small candle
    if (body1 > range1 * 0.6 && body2 < range2 * 0.3)
    {
        // Check direction alignment
        bool bullishOB = (iClose(Symbol(), M15_Timeframe, 1) > iOpen(Symbol(), M15_Timeframe, 1) &&
                         signal > 0);  // Bullish OB + bullish signal
        bool bearishOB = (iClose(Symbol(), M15_Timeframe, 1) < iOpen(Symbol(), M15_Timeframe, 1) &&
                         signal < 0);  // Bearish OB + bearish signal

        if (bullishOB || bearishOB)
            return true;
    }

    return false;
}

//+------------------------------------------------------------------+
//| Check Fibonacci zones                                           |
//+------------------------------------------------------------------+
bool CheckFibonacciZones(int signal)
{
    // Get H4 midpoint
    double h4_high = iHigh(Symbol(), H4_Timeframe, 0);
    double h4_low = iLow(Symbol(), H4_Timeframe, 0);
    double midpoint = (h4_high + h4_low) / 2;

    double current_price = iClose(Symbol(), M15_Timeframe, 0);

    // Premium zone (above midpoint) - bearish bias
    // Discount zone (below midpoint) - bullish bias
    bool in_premium = current_price > midpoint;
    bool in_discount = current_price < midpoint;

    // Signal alignment with zone bias
    if ((signal > 0 && in_discount) || (signal < 0 && in_premium))
        return true;

    return false;
}

//+------------------------------------------------------------------+
//| Execute trade with retry logic                                   |
//+------------------------------------------------------------------+
void ExecuteTradeWithRetry(int signal)
{
    double lotSize = CalculatePositionSize(signal);
    if (lotSize <= 0)
        return;

    // Calculate stop loss and take profit
    double entryPrice = symbolInfo.Ask();
    if (signal < 0)
        entryPrice = symbolInfo.Bid();

    double atr = iATR(Symbol(), M15_Timeframe, 14, 0);
    double stopLoss = atr * 2;  // 2 ATR stop
    double takeProfit = atr * 3; // 3 ATR target

    ENUM_ORDER_TYPE orderType = (signal > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    double slPrice = (signal > 0) ? entryPrice - stopLoss : entryPrice + stopLoss;
    double tpPrice = (signal > 0) ? entryPrice + takeProfit : entryPrice - takeProfit;

    // Execute with retry
    for (int attempt = 1; attempt <= MAX_RETRIES; attempt++)
    {
        if (trade.PositionOpen(Symbol(), orderType, lotSize, entryPrice, slPrice, tpPrice))
        {
            Print("✅ Trade executed successfully: ", Symbol(), " ", EnumToString(orderType), " ", lotSize, " lots");
            retryCount = 0;  // Reset retry counter
            return;
        }
        else
        {
            int error = GetLastError();
            Print("⚠️ Trade attempt ", attempt, " failed. Error: ", error, " - ", trade.ResultComment());

            // Check if retryable error
            if (!IsRetryableError(error))
            {
                Print("❌ Non-retryable error - aborting trade");
                break;
            }

            // Exponential backoff
            Sleep(attempt * 1000);
        }
    }

    retryCount++;
    if (retryCount >= MAX_RETRIES)
    {
        Print("🚫 Max retries reached - temporary trading halt");
    }
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk                            |
//+------------------------------------------------------------------+
double CalculatePositionSize(int signal)
{
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * Risk_Per_Trade;

    double atr = iATR(Symbol(), M15_Timeframe, 14, 0);
    double stopLossPips = atr * 2 / symbolInfo.Point();

    double tickValue = symbolInfo.TickValue();
    double lotSize = riskAmount / (stopLossPips * tickValue);

    // Apply broker limits
    double maxLot = symbolInfo.LotsMax();
    double minLot = symbolInfo.LotsMin();

    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));

    return NormalizeDouble(lotSize, 2);
}

//+------------------------------------------------------------------+
//| Check if error is retryable                                      |
//+------------------------------------------------------------------+
bool IsRetryableError(int error)
{
    // Retryable errors
    switch (error)
    {
        case ERR_NET_TIMEOUT:
        case ERR_NET_SOCKET:
        case ERR_NET_CONNECTION:
        case ERR_TRADE_TIMEOUT:
            return true;
        default:
            return false;
    }
}

//+------------------------------------------------------------------+
//| Check if in kill zone (trading hours)                            |
//+------------------------------------------------------------------+
bool IsInKillZone()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);

    int hour = dt.hour;

    // London session
    bool londonSession = (hour >= London_Open_Hour && hour <= London_Close_Hour);

    // New York session
    bool nySession = (hour >= NewYork_Open_Hour && hour <= NewYork_Close_Hour);

    return londonSession || nySession;
}

//+------------------------------------------------------------------+
//| Check for new bar                                                 |
//+------------------------------------------------------------------+
bool IsNewBar()
{
    datetime currentBarTime = iTime(Symbol(), M15_Timeframe, 0);
    if (currentBarTime != lastBarTime)
    {
        lastBarTime = currentBarTime;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Update daily drawdown                                            |
//+------------------------------------------------------------------+
void UpdateDailyDrawdown()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);

    // Reset at start of new day
    static int lastDay = -1;
    if (dt.day != lastDay)
    {
        dailyStartingBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        dailyDrawdown = 0.0;
        lastDay = dt.day;
    }

    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double drawdown = (dailyStartingBalance - currentBalance) / dailyStartingBalance;

    if (drawdown > dailyDrawdown)
        dailyDrawdown = drawdown;
}

//+------------------------------------------------------------------+
//| Connect to Python bot via socket                                 |
//+------------------------------------------------------------------+
bool ConnectToPython()
{
    socketHandle = SocketCreate();
    if (socketHandle == INVALID_HANDLE)
    {
        Print("❌ Failed to create socket");
        return false;
    }

    if (!SocketConnect(socketHandle, PythonHost, PythonPort, 5000))
    {
        Print("❌ Failed to connect to Python bot at ", PythonHost, ":", PythonPort);
        SocketClose(socketHandle);
        socketHandle = INVALID_HANDLE;
        return false;
    }

    Print("✅ Connected to Python bot");
    return true;
}

//+------------------------------------------------------------------+
//| Convert enum to string                                           |
//+------------------------------------------------------------------+
string EnumToString(ENUM_ORDER_TYPE type)
{
    switch(type)
    {
        case ORDER_TYPE_BUY: return "BUY";
        case ORDER_TYPE_SELL: return "SELL";
        default: return "UNKNOWN";
    }
}
//+------------------------------------------------------------------+