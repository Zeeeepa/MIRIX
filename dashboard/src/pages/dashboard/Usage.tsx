import { useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Coins, AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const Usage: React.FC = () => {
  const { user, refreshUser } = useAuth();

  const credits = user?.credits ?? 0;

  useEffect(() => {
    refreshUser();

    const interval = setInterval(() => {
      refreshUser();
    }, 30000);

    return () => clearInterval(interval);
  }, [refreshUser]);

  const getStatusColor = () => {
    if (credits <= 0) return 'text-red-500';
    if (credits < 1) return 'text-orange-500';
    if (credits < 5) return 'text-yellow-500';
    return 'text-emerald-500';
  };

  const formatCredits = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    }).format(value);
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Usage & Billing</h2>
          <p className="text-muted-foreground">Monitor your API usage and credit balance.</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refreshUser()}
          className="gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      <Card className="overflow-hidden">
        <div className="bg-gradient-to-br from-violet-500/10 via-purple-500/5 to-fuchsia-500/10">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-lg font-medium text-muted-foreground">
                  Available Credits
                </CardTitle>
                <div className={`text-5xl font-bold mt-2 ${getStatusColor()}`}>
                  {formatCredits(credits)}
                </div>
              </div>
              <div
                className={`p-4 rounded-full ${credits <= 0 ? 'bg-red-500/10' : 'bg-emerald-500/10'}`}
              >
                {credits <= 0 ? (
                  <AlertTriangle className="h-8 w-8 text-red-500" />
                ) : (
                  <Coins className={`h-8 w-8 ${getStatusColor()}`} />
                )}
              </div>
            </div>
          </CardHeader>
        </div>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Credit Status</CardTitle>
          <div className={`h-3 w-3 rounded-full ${getStatusColor().replace('text-', 'bg-')}`} />
        </CardHeader>
        <CardContent>
          <div className={`text-2xl font-bold ${getStatusColor()}`}>
            {credits <= 0 ? 'Depleted' : credits < 1 ? 'Low' : credits < 5 ? 'Moderate' : 'Healthy'}
          </div>
          <p className="text-xs text-muted-foreground">
            {credits <= 0 ? 'Contact support for more credits' : 'Credits available for use'}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">About Credits</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            1 credit = $1. Costs are calculated based on model-specific token pricing.
          </p>
        </CardContent>
      </Card>
    </div>
  );
};
