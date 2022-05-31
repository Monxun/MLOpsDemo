from ninja import NinjaAPI

# INSTANTIATE API OBJECT FROM DJANGO-NINJA
api = NinjaAPI(version='stocks')


# API ENDPOINTS FOR tiingo.py
from .src.tiingo import get_meta_data, get_price_data

@api.get("/meta/{tid}")
def GetMetaData(request, tid: str):
    return get_meta_data(tid)


@api.get("/price/{tid}")
def GetPriceData(request, tid: str):
    return get_price_data(tid)