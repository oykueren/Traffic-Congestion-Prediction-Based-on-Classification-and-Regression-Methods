import pyproj

def convert_to_utm(df, src_epsg, dst_epsg, col_lat, col_lon, alias_lon=None,
                   alias_lat=None):
    df2 = df
    old_proj = pyproj.Proj(src_epsg, preserve_units=True)
    new_proj = pyproj.Proj(dst_epsg, preserve_units=True)
    print("Formal definition string for the old projection:",
          old_proj.definition_string())
    print("Formal definition string for the new projection:",
          new_proj.definition_string())
    lon = df[col_lon].values
    lat = df[col_lat].values
    x1, y1 = old_proj(lon, lat)

    x2, y2 = pyproj.transform(old_proj, new_proj, x1, y1)

    if alias_lon is None:
        alias_lon = col_lon

    if alias_lat is None:
        alias_lat = col_lat

    df[alias_lon] = x2
    df[alias_lat] = y2

    return df