import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from stock_data import StockData


class Visualizer:
    @staticmethod
    def create_stock_price_chart(data, ticker):
        """Create an enhanced interactive stock price chart with updated strategy signals"""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f'{ticker} Price Action with Strategy Signals',
                'Weekly MACD (5,8) Indicator'
            )
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350',
                showlegend=True
            ),
            row=1, col=1
        )

        # Add weekly 50 EMA with prominence
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Weekly_EMA_50'],
                name='Weekly 50 EMA',
                line=dict(color='#FFB74D', width=2),
                opacity=0.9
            ),
            row=1, col=1
        )

        # Add buy signals with enhanced visibility and information
        buy_signals = data[data['Buy_Signal']]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Low'] * 0.99,
                    mode='markers+text',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#66BB6A',
                        line=dict(color='#2E7D32', width=2)
                    ),
                    text=['BUY'] * len(buy_signals),
                    textposition='bottom center',
                    textfont=dict(size=10, color='#2E7D32'),
                    hovertemplate=(
                        "Buy Signal<br>" +
                        "Date: %{x}<br>" +
                        "Price: %{y:.2f}<br>" +
                        "Conditions Met:<br>" +
                        "• Price > Weekly 50 EMA<br>" +
                        "• MACD Crossover Below Zero<extra></extra>"
                    )
                ),
                row=1, col=1
            )

        # Add MACD lines
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='Weekly MACD (5,8)',
                line=dict(color='#42A5F5', width=2)
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Signal_Line'],
                name='Signal (5)',
                line=dict(color='#FFB74D', width=2)
            ),
            row=2, col=1
        )

        # Add MACD histogram with enhanced colors
        colors = ['#26A69A' if val >= 0 else '#EF5350' for val in data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

        # Add zero line for MACD with more prominence
        fig.add_hline(
            y=0,
            line_width=1,
            line_dash="solid",
            line_color="rgba(255, 255, 255, 0.5)",
            row=2, col=1,
            annotation=dict(
                text="Zero Line",
                xref="paper",
                x=1.02,
                showarrow=False,
                font=dict(size=10, color="rgba(255, 255, 255, 0.5)")
            )
        )

        # Update layout with improved design and auto-scaling
        fig.update_layout(
            template='plotly_dark',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            margin=dict(t=30, l=50, r=50),
            xaxis_autorange=True,
            yaxis_autorange=True,
            yaxis2_autorange=True
        )

        # Update x-axes
        fig.update_xaxes(
            gridcolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="rgba(0,0,0,0.5)",
            ),
            autorange=True,
            type="date"
        )

        # Update y-axes
        fig.update_yaxes(
            gridcolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            title_text="Price",
            row=1, col=1,
            autorange=True,
            fixedrange=False,
            rangemode="tozero"
        )

        fig.update_yaxes(
            gridcolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            title_text="MACD",
            row=2, col=1,
            autorange=True,
            fixedrange=False,
            rangemode="normal"
        )

        # Add hover and zoom settings
        fig.update_layout(
            uirevision='dataset',
            hovermode='x unified',
            dragmode='zoom',
            selectdirection='h',
            autosize=True
        )

        return fig
    @staticmethod
    def create_portfolio_weights_chart(weights, tickers):
        """Create a pie chart to visualize portfolio weights"""
        import plotly.graph_objects as go
        
        # Format weights as percentages for display
        weights_pct = [weight * 100 for weight in weights]
        
        # Create hover text with both percentage and ticker
        hover_text = [f"{ticker}: {weight:.2f}%" for ticker, weight in zip(tickers, weights_pct)]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=tickers,
            values=weights,
            textinfo='label+percent',
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
                line=dict(color='#000000', width=0.5)
            ),
            textfont=dict(size=14),
            insidetextorientation='radial'
        )])
        
        # Update layout for better appearance
        fig.update_layout(
            title_text="Portfolio Allocation",
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            height=500,
            margin=dict(t=30, l=30, r=30, b=30)
        )
        
        return fig

    @staticmethod
    def format_number(number):
        """Format numbers for better readability"""
        if number >= 1e6:
            return f'${number/1e6:.1f}M'
        elif number >= 1e3:
            return f'${number/1e3:.1f}K'
        else:
            return f'${number:.2f}'

    @staticmethod
    def create_metrics_table(metrics):
        """Create formatted metrics table"""
        df = pd.DataFrame(metrics).T
        df.columns = ['Value']
        return df
