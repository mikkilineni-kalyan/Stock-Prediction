from flask import Blueprint, request, jsonify

import pandas as pd

import os

import logging



search_bp = Blueprint('search', __name__)

logger = logging.getLogger(__name__)



@search_bp.route('/api/search', methods=['GET'])

def search_stocks():

    query = request.args.get('query', '').upper()

    if not query:

        return jsonify([])



    try:

        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nasdaq-listed.csv')

        

        if not os.path.exists(csv_path):

            logger.error(f"CSV file not found at: {csv_path}")

            return jsonify({'error': 'Stock data not available'}), 500

        

        df = pd.read_csv(csv_path)

        

        # Fill NaN values

        df = df.fillna('')

        

        matches = df[

            df['Symbol'].str.contains(query, case=False, na=False) |

            df['Name'].str.contains(query, case=False, na=False)

        ]

        

        results = matches.head(20).apply(

            lambda x: {

                'symbol': str(x['Symbol']),

                'name': str(x['Name']),

                'sector': str(x['Sector']),

                'industry': str(x['Industry'])

            }, 

            axis=1

        ).tolist()

        

        return jsonify(results)



    except Exception as e:

        logger.error(f"Search error: {str(e)}")

        return jsonify({'error': str(e)}), 500
