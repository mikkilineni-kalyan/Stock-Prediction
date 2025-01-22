import { Subject } from 'rxjs';

interface Pattern {
    name: string;
    confidence: number;
}

interface AlertData {
    patterns: Pattern[];
    ticker: string;
}

interface AlertRule {
    condition: (data: AlertData) => boolean;
    createAlert: (ticker: string) => Alert;
}

interface Alert {
    type: string;
    severity: string;
    message: string;
    timestamp: Date;
}

class AlertService {
    private rules: Map<string, AlertRule> = new Map();
    private alerts = new Subject<Alert>();

    constructor() {
        this.initializeRules();
    }

    private initializeRules() {
        this.addRule('headAndShoulders', {
            condition: (data: AlertData) => 
                data.patterns.some(p => 
                    p.name === 'Head and Shoulders' && p.confidence > 0.8
                ),
            createAlert: (ticker: string) => ({
                type: 'pattern',
                severity: 'warning',
                message: `Head and Shoulders pattern detected for ${ticker}`,
                timestamp: new Date()
            })
        });
    }

    private addRule(name: string, rule: AlertRule) {
        this.rules.set(name, rule);
    }

    public checkAlerts(data: AlertData) {
        this.rules.forEach((rule, name) => {
            if (rule.condition(data)) {
                const alert = rule.createAlert(data.ticker);
                this.alerts.next(alert);
            }
        });
    }

    public getAlerts() {
        return this.alerts.asObservable();
    }
}

export default new AlertService(); 