'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent } from '@/components/ui/card';

export default function ManagersRedirect() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to funds page
    router.replace('/funds');
  }, [router]);

  return (
    <div className="flex items-center justify-center h-64">
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-muted-foreground">
            Redirecting to Funds...
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            The system has been updated to use collaborative funds instead of individual managers.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
