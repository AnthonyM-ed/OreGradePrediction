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

class HoleSamplesElementGrades(models.Model):
    """
    Model for the new view vw_HoleSamples_ElementGrades
    This view combines assay, sample, and collar data with standardized grades in PPM
    """
    sample_id = models.CharField(max_length=100, db_column='SampleID', primary_key=True)
    hole_id = models.CharField(max_length=50, db_column='Hole_ID')
    dataset = models.CharField(max_length=50, db_column='DataSet')
    element = models.CharField(max_length=10, db_column='Element')
    standardized_grade_ppm = models.FloatField(db_column='standardized_grade_ppm', null=True, blank=True)
    depth_from = models.FloatField(db_column='Depth_From', null=True, blank=True)
    depth_to = models.FloatField(db_column='Depth_To', null=True, blank=True)
    interval_length = models.FloatField(db_column='Interval_Length', null=True, blank=True)
    latitude = models.FloatField(db_column='latitude', null=True, blank=True)
    longitude = models.FloatField(db_column='longitude', null=True, blank=True)
    elevation = models.FloatField(db_column='elevation', null=True, blank=True)
    lab_code = models.CharField(max_length=20, db_column='LabCode', null=True, blank=True)
    
    class Meta:
        db_table = 'vw_HoleSamples_ElementGrades'
        managed = False  # Django won't manage this view
        
    def __str__(self):
        return f"{self.sample_id} - {self.element}: {self.standardized_grade_ppm} ppm"
    
    @property
    def mid_depth(self):
        """Calculate middle depth of the interval"""
        if self.depth_from is not None and self.depth_to is not None:
            return (self.depth_from + self.depth_to) / 2
        return None
    
    @property
    def grade_tonnage(self):
        """Calculate grade tonnage proxy"""
        if self.standardized_grade_ppm is not None and self.interval_length is not None:
            return self.standardized_grade_ppm * self.interval_length
        return None