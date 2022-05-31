from django import forms

class TickerForm(forms.Form):

    # views.py> ...ticker = request.POST['ticker']
    ticker = forms.CharField(label='Ticker', max_length=5)
