from scopy.sunlight import SiteData


location_name, location_latitude, location_longitude = 'Jiexi County', 23.45, 115.90

loc_data = SiteData(name=location_name,
                    latitude=location_latitude,
                    longitude=location_longitude)

print(loc_data.dni_sum / 1000)
