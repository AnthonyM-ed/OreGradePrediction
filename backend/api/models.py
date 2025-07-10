from django.db import models

class HoleGradesMapData(models.Model):
    """
    Modelo para la tabla tblHoleGrades_MapData_Standardized_Cache
    """
    hole_id = models.CharField(max_length=50, db_column='Hole_ID')
    element = models.CharField(max_length=10, db_column='Element')
    dataset = models.CharField(max_length=50, db_column='DataSet')
    weighted_grade = models.FloatField(db_column='weighted_grade')
    latitude = models.FloatField(db_column='latitude', null=True, blank=True)
    longitude = models.FloatField(db_column='longitude', null=True, blank=True)
    
    class Meta:
        db_table = 'tblHoleGrades_MapData_Standardized_Cache'
        managed = False  # Django no gestionará esta tabla

class DHColl(models.Model):
    """
    Modelo para la tabla tblDHColl
    """
    hole_id = models.CharField(max_length=50, db_column='Hole_ID', primary_key=True)
    ll_lat = models.FloatField(db_column='LL_Lat', null=True, blank=True)
    ll_long = models.FloatField(db_column='LL_Long', null=True, blank=True)
    
    class Meta:
        db_table = 'tblDHColl'
        managed = False  # Django no gestionará esta tabla