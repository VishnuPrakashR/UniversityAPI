#  Copyright (c) 2023. This is the property of Vishnu Prakash
import json
import datetime
from bson import ObjectId


# Encodes to into json
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):     # Convert if ObjectId
            return str(o)
        if isinstance(o, (datetime.date, datetime.datetime)):   # Convert if date time
            return o.isoformat()
        return json.JSONEncoder.default(self, o)