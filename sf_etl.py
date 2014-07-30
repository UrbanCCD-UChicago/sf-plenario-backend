from wopr import create_app
from wopr.etl import import_shapefile, create_meta_table, add_dataset_meta
#celery_app = make_celery(app=app)

proj_str = '+proj=lcc +lat_1=37.06666666666667 +lat_2=38.43333333333333 \
+lat_0=36.5 +lon_0=-120.5 +x_0=2000000 +y_0=500000.0000000002 +datum=NAD83 \
+units=us-ft +no_defs'

app = create_app()

if __name__ == "__main__":
    #sf_raw_crime(fpath='data/sfpd_incident_all_csv.zip')
    #sf_dat_crime(fpath='data/sfpd_incident_all_csv.zip', crime_type='violent')
    #sf_dat_crime(fpath='data/sfpd_incident_all_csv.zip', crime_type='property')
    #import_shapefile('./data/sf_census_blocks.zip', 'sf_census_blocks', proj=4326)
    #import_shapefile('./data/CI.2.d.BlockParties.zip', 'sf_block_parties',
    #    proj=proj_str, voronoi=False)
    #import_shapefile('./data/CI.1.a.CommunityCenterAccess.zip',
    #    'sf_community_centers', proj=proj_str)
    #import_shapefile('./data/ED.2.a.EmploymentRate.zip',
    #    'sf_employment_rate', proj=proj_str)
    #import_shapefile('./data/sf_open_space.zip', 'sf_open_space', proj=proj_str)
    #import_shapefile('./data/HWB.2.a.FarmersMarketAccess.zip', 'sf_farmers_markets',
    #    proj=proj_str)
    #import_shapefile('./data/sf_building_footprint.zip', 'sf_building_footprint',
    #    proj=proj_str)
    #import_shapefile('./data/CI.1.b.NeighborhoodCommercialZoning.zip', 'sf_commercial_zoning',
    #    proj=proj_str)
    #import_shapefile('./data/sf_building_footprint.zip', 'sf_building_footprint',
    #    proj=proj_str)
    #import_shapefile('./data/sfshoreext.zip', 'sf_shore', proj=proj_str)
    #import_shapefile('./data/SF_Urban_Tree_Canopy.zip', 'sf_tree_canopy', proj=proj_str)
    create_meta_table()
    add_dataset_meta('sf_open_space', human_name='Open space', area_q=True, dist_q=True)
    add_dataset_meta('sf_building_footprint', human_name='Building footprint',
        area_q=True, dist_q=True)
    add_dataset_meta('sf_tree_canopy', human_name='Tree canopy', area_q=True)
    add_dataset_meta('sf_block_parties', human_name='Block parties',
        count_q=True, dist_q=True)
    add_dataset_meta('sf_community_centers', human_name='Community centers',
        count_q=True, dist_q=True)
    add_dataset_meta('sf_employment_rate', human_name='Employment rate',
        weighted_q=True, val_attr='employ_pct')
