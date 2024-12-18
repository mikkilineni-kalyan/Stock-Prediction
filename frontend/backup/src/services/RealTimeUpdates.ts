import { io, Socket } from 'socket.io-client';
import { Subject } from 'rxjs';

export interface MarketUpdate {
    ticker: string;
    price: number;
    volume: number;
    timestamp: string;
}

class RealTimeService {
    private socket: Socket;
    private marketUpdates = new Subject<MarketUpdate>();

    constructor() {
        this.socket = io('http://localhost:5000');
        
        this.socket.on('connect', () => {
            console.log('Connected to WebSocket server');
        });

        this.socket.on('market_update', (update: MarketUpdate) => {
            this.marketUpdates.next(update);
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from WebSocket server');
        });
    }

    subscribeToUpdates(ticker: string) {
        this.socket.emit('subscribe', { ticker });
    }

    unsubscribeFromUpdates(ticker: string) {
        this.socket.emit('unsubscribe', { ticker });
    }

    getMarketUpdates() {
        return this.marketUpdates.asObservable();
    }
}

export const realTimeService = new RealTimeService(); 