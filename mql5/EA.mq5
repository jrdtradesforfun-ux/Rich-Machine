//+------------------------------------------------------------------+
//|                    Universal Trading Bot EA                      |
//|                        MetaTrader 5 Expert Advisor                |
//|                    Communicates with Python Trading Bot           |
//+------------------------------------------------------------------+

#property copyright "Universal Trading Bot"
#property link      "https://github.com/universal-trading-bot"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>

//--- Input parameters
input string PythonHost = "localhost";    // Python bot host
input int PythonPort = 5000;              // Python bot port
input int UpdateInterval = 1000;          // Update interval in milliseconds

//--- Global variables
int socketHandle = INVALID_HANDLE;
CTrade trade;
CSymbolInfo symbolInfo;
CPositionInfo positionInfo;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("Universal Trading Bot EA initializing...");
    
    // Initialize trade object
    trade.SetExpertMagicNumber(123456);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(Symbol());
    
    // Initialize symbol info
    symbolInfo.Name(Symbol());
    
    // Connect to Python bot
    if (!ConnectToPython())
    {
        Print("Failed to connect to Python bot");
        return INIT_FAILED;
    }
    
    Print("EA initialized successfully");
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
    
    Print("EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    static datetime lastUpdate = 0;
    
    // Update every UpdateInterval milliseconds
    if (TimeCurrent() - lastUpdate < UpdateInterval / 1000)
        return;
        
    lastUpdate = TimeCurrent();
    
    // Process commands from Python bot
    ProcessPythonCommands();
    
    // Send market data to Python bot
    SendMarketData();
}

//+------------------------------------------------------------------+
//| Connect to Python bot via socket                                 |
//+------------------------------------------------------------------+
bool ConnectToPython()
{
    socketHandle = SocketCreate();
    if (socketHandle == INVALID_HANDLE)
    {
        Print("Failed to create socket");
        return false;
    }
    
    if (!SocketConnect(socketHandle, PythonHost, PythonPort, 5000))
    {
        Print("Failed to connect to Python bot at ", PythonHost, ":", PythonPort);
        SocketClose(socketHandle);
        socketHandle = INVALID_HANDLE;
        return false;
    }
    
    Print("Connected to Python bot");
    return true;
}

//+------------------------------------------------------------------+
//| Process commands from Python bot                                 |
//+------------------------------------------------------------------+
void ProcessPythonCommands()
{
    // Check if socket is valid
    if (socketHandle == INVALID_HANDLE)
        return;
        
    // Check for incoming data
    uint bytesAvailable = SocketBytesAvailable(socketHandle);
    if (bytesAvailable == 0)
        return;
        
    // Read data
    uchar buffer[];
    ArrayResize(buffer, bytesAvailable);
    
    int bytesRead = SocketRead(socketHandle, buffer, bytesAvailable, 5000);
    if (bytesRead <= 0)
        return;
        
    // Convert to string
    string receivedData = CharArrayToString(buffer);
    
    // Parse JSON command
    // Note: MQL5 doesn't have built-in JSON parsing, simplified parsing here
    if (StringFind(receivedData, "place_order") >= 0)
    {
        ProcessPlaceOrderCommand(receivedData);
    }
    else if (StringFind(receivedData, "close_position") >= 0)
    {
        ProcessClosePositionCommand(receivedData);
    }
    else if (StringFind(receivedData, "get_balance") >= 0)
    {
        SendAccountBalance();
    }
    else if (StringFind(receivedData, "get_positions") >= 0)
    {
        SendPositions();
    }
    else if (StringFind(receivedData, "get_market_data") >= 0)
    {
        SendMarketData();
    }
}

//+------------------------------------------------------------------+
//| Process place order command                                      |
//+------------------------------------------------------------------+
void ProcessPlaceOrderCommand(string command)
{
    // Simplified parsing - in production use proper JSON parser
    string symbol = ExtractValue(command, "symbol");
    string orderType = ExtractValue(command, "type");
    double volume = StringToDouble(ExtractValue(command, "volume"));
    double price = StringToDouble(ExtractValue(command, "price"));
    double sl = StringToDouble(ExtractValue(command, "sl"));
    double tp = StringToDouble(ExtractValue(command, "tp"));
    
    // Validate symbol
    if (symbol != Symbol())
    {
        SendResponse("{\"success\": false, \"error\": \"Symbol mismatch\"}");
        return;
    }
    
    // Determine order type
    ENUM_ORDER_TYPE order_type;
    if (orderType == "buy")
        order_type = ORDER_TYPE_BUY;
    else if (orderType == "sell")
        order_type = ORDER_TYPE_SELL;
    else
    {
        SendResponse("{\"success\": false, \"error\": \"Invalid order type\"}");
        return;
    }
    
    // Place order
    bool result = trade.PositionOpen(symbol, order_type, volume, price, sl, tp);
    
    if (result)
    {
        SendResponse("{\"success\": true, \"ticket\": " + IntegerToString(trade.ResultOrder()) + "}");
        Print("Order placed successfully: ", symbol, " ", orderType, " ", volume);
    }
    else
    {
        SendResponse("{\"success\": false, \"error\": \"" + trade.ResultComment() + "\"}");
        Print("Order failed: ", trade.ResultComment());
    }
}

//+------------------------------------------------------------------+
//| Process close position command                                   |
//+------------------------------------------------------------------+
void ProcessClosePositionCommand(string command)
{
    int ticket = (int)StringToInteger(ExtractValue(command, "ticket"));
    
    // Find and close position
    if (PositionSelectByTicket(ticket))
    {
        bool result = trade.PositionClose(ticket);
        
        if (result)
        {
            SendResponse("{\"success\": true}");
            Print("Position closed: ", ticket);
        }
        else
        {
            SendResponse("{\"success\": false, \"error\": \"" + trade.ResultComment() + "\"}");
        }
    }
    else
    {
        SendResponse("{\"success\": false, \"error\": \"Position not found\"}");
    }
}

//+------------------------------------------------------------------+
//| Send account balance                                             |
//+------------------------------------------------------------------+
void SendAccountBalance()
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    string response = "{\"balance\": " + DoubleToString(balance, 2) + 
                     ", \"equity\": " + DoubleToString(equity, 2) + "}";
    SendResponse(response);
}

//+------------------------------------------------------------------+
//| Send positions information                                       |
//+------------------------------------------------------------------+
void SendPositions()
{
    string positions = "[";
    
    for (int i = 0; i < PositionsTotal(); i++)
    {
        if (PositionGetSymbol(i) == Symbol())
        {
            if (i > 0) positions += ",";
            
            positions += "{";
            positions += "\"ticket\": " + IntegerToString(PositionGetInteger(POSITION_TICKET)) + ",";
            positions += "\"symbol\": \"" + PositionGetString(POSITION_SYMBOL) + "\",";
            positions += "\"type\": \"" + EnumToString((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE)) + "\",";
            positions += "\"volume\": " + DoubleToString(PositionGetDouble(POSITION_VOLUME), 2) + ",";
            positions += "\"price\": " + DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), 5) + ",";
            positions += "\"sl\": " + DoubleToString(PositionGetDouble(POSITION_SL), 5) + ",";
            positions += "\"tp\": " + DoubleToString(PositionGetDouble(POSITION_TP), 5) + ",";
            positions += "\"profit\": " + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2);
            positions += "}";
        }
    }
    
    positions += "]";
    SendResponse("{\"positions\": " + positions + "}");
}

//+------------------------------------------------------------------+
//| Send market data                                                 |
//+------------------------------------------------------------------+
void SendMarketData()
{
    string symbol = Symbol();
    
    // Get current prices
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
    
    // Get OHLC
    double open = iOpen(symbol, PERIOD_CURRENT, 0);
    double high = iHigh(symbol, PERIOD_CURRENT, 0);
    double low = iLow(symbol, PERIOD_CURRENT, 0);
    double close = iClose(symbol, PERIOD_CURRENT, 0);
    long volume = iVolume(symbol, PERIOD_CURRENT, 0);
    
    string response = "{";
    response += "\"symbol\": \"" + symbol + "\",";
    response += "\"bid\": " + DoubleToString(bid, 5) + ",";
    response += "\"ask\": " + DoubleToString(ask, 5) + ",";
    response += "\"spread\": " + IntegerToString(spread) + ",";
    response += "\"open\": " + DoubleToString(open, 5) + ",";
    response += "\"high\": " + DoubleToString(high, 5) + ",";
    response += "\"low\": " + DoubleToString(low, 5) + ",";
    response += "\"close\": " + DoubleToString(close, 5) + ",";
    response += "\"volume\": " + IntegerToString(volume);
    response += "}";
    
    SendResponse("{\"market_data\": " + response + "}");
}

//+------------------------------------------------------------------+
//| Send response to Python bot                                      |
//+------------------------------------------------------------------+
void SendResponse(string response)
{
    if (socketHandle == INVALID_HANDLE)
        return;
        
    // Convert string to char array
    uchar buffer[];
    StringToCharArray(response, buffer);
    
    // Send data
    SocketSend(socketHandle, buffer, ArraySize(buffer));
}

//+------------------------------------------------------------------+
//| Extract value from JSON-like string (simplified)                |
//+------------------------------------------------------------------+
string ExtractValue(string json, string key)
{
    string search = "\"" + key + "\":";
    int start = StringFind(json, search);
    
    if (start == -1)
        return "";
        
    start += StringLen(search);
    
    // Find end of value
    int end = start;
    int braceCount = 0;
    bool inString = false;
    
    for (int i = start; i < StringLen(json); i++)
    {
        char ch = StringGetCharacter(json, i);
        
        if (ch == '"' && (i == 0 || StringGetCharacter(json, i-1) != '\\'))
            inString = !inString;
        else if (!inString)
        {
            if (ch == '{' || ch == '[')
                braceCount++;
            else if (ch == '}' || ch == ']')
                braceCount--;
            else if (ch == ',' && braceCount == 0)
            {
                end = i;
                break;
            }
        }
    }
    
    if (end == start)
        end = StringLen(json);
        
    string value = StringSubstr(json, start, end - start);
    StringTrimLeft(value);
    StringTrimRight(value);
    
    // Remove quotes if present
    if (StringGetCharacter(value, 0) == '"')
        value = StringSubstr(value, 1, StringLen(value) - 2);
        
    return value;
}

//+------------------------------------------------------------------+
//| Convert enum to string                                           |
//+------------------------------------------------------------------+
string EnumToString(ENUM_POSITION_TYPE type)
{
    switch(type)
    {
        case POSITION_TYPE_BUY: return "buy";
        case POSITION_TYPE_SELL: return "sell";
        default: return "unknown";
    }
}
//+------------------------------------------------------------------+